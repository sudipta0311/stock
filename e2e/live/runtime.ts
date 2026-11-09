import AxeBuilder from "@axe-core/playwright";
import { chromium, test as base } from "@playwright/test";
import type {
  BrowserContext,
  Page,
  Request,
  Route,
  Response,
} from "@playwright/test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { getLiveEnvironment } from "./environment.mjs";
import { decideRequest, routeTemplate } from "./request-guard";

type Severity = "fail" | "note";
type CoverageField = "expanders" | "dialogs" | "listItems";

export interface Finding {
  severity: Severity;
  screen: string;
  check: string;
  detail: string;
  screenshot: string;
}

interface Coverage {
  screen: string;
  expanders: number;
  dialogs: number;
  listItems: number;
}

interface LiveEnvironment {
  profileDir: string;
  outDir: string;
  pwaUrl: string;
  apiUrl: string;
  chromePath?: string;
  headed: boolean;
}

export interface LiveRuntime {
  page: Page;
  environment: LiveEnvironment;
  findings: Finding[];
  snapshots: Map<string, unknown>;
  setScreen(path: string): void;
  addFinding(severity: Severity, check: string, detail: string): void;
  count(field: CoverageField, amount?: number): void;
  checkScreen(startedAt?: number): Promise<void>;
  screenshot(): Promise<void>;
  writeReports(): Promise<{ fails: number; notes: number }>;
}

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const apiWrites = new Set(["POST", "PUT", "PATCH", "DELETE"]);
interface BannedTerm {
  term: string;
  allowed_context?: string;
}

function escapeRegExp(value: string): string {
  const special = new Set([
    ".",
    "*",
    "+",
    "?",
    "^",
    "$",
    "(",
    ")",
    "|",
    "[",
    "]",
    "{",
    "}",
    "\\",
  ]);
  return [...value]
    .map((character) => (special.has(character) ? "\\" + character : character))
    .join("");
}

function normalized(value: string): string {
  return value.replace(/[’‘]/g, "'").replace(/\s+/g, " ").toLowerCase();
}

function evidenceWindow(
  text: string,
  index: number,
  matchLength: number,
  maximum = 120,
): string {
  const start = Math.max(0, index);
  const length = Math.max(1, Math.min(matchLength, maximum));
  let remaining = maximum - length;
  let before = Math.min(start, Math.floor(remaining / 2));
  remaining -= before;
  let after = Math.min(text.length - (start + length), remaining);
  remaining -= after;
  before += Math.min(start - before, remaining);
  return text.slice(start - before, start + length + after);
}

function evidenceDetail(
  text: string,
  index: number,
  matchLength: number,
  label = "",
): string {
  const prefix = label ? label + ": " : "";
  return (
    prefix +
    evidenceWindow(text, index, matchLength, Math.max(1, 120 - prefix.length))
  );
}

function safeOutput(value: string, environment: LiveEnvironment): string {
  let result = value;
  for (const configured of [
    environment.profileDir,
    environment.outDir,
    environment.pwaUrl,
    environment.apiUrl,
    environment.chromePath,
    process.env.ProgramFiles,
    process.env["ProgramFiles(x86)"],
    process.env.LOCALAPPDATA,
    process.env.USERPROFILE,
    process.env.TEMP,
  ]) {
    if (configured) {
      result = result.replace(
        new RegExp(escapeRegExp(configured), "gi"),
        "[configured value]",
      );
    }
  }
  return result
    .replace(/[\r\n]+/g, " ")
    .replace(/postgres(?:ql)?:\/\/[^\s)]+/gi, "[database URL]")
    .replace(/\b[A-Z]{5}[0-9]{4}[A-Z]\b/gi, "[PAN redacted]")
    .replace(/Bearer\s+[^\s]+/gi, "Bearer [redacted]");
}

function firstLine(value: string): string {
  return value.split(/\r?\n/, 1)[0]?.trim() ?? "Unknown error.";
}

async function bannedTerms(): Promise<BannedTerm[]> {
  const source = await readFile(
    resolve(repositoryRoot, "apps/web/src/copy/banned-terms.json"),
    "utf8",
  );
  const data = JSON.parse(source) as { terms?: BannedTerm[] };
  return (data.terms ?? []).filter(
    (entry) => typeof entry.term === "string" && entry.term.length > 0,
  );
}

async function copyKeyPaths(): Promise<Set<string>> {
  const source = await readFile(
    resolve(repositoryRoot, "apps/web/src/copy/en.json"),
    "utf8",
  );
  const data: unknown = JSON.parse(source);
  const paths = new Set<string>();
  const visit = (value: unknown, prefix: string): void => {
    if (Array.isArray(value)) {
      value.forEach((item, index) => visit(item, `${prefix}.${index}`));
      return;
    }
    if (typeof value === "object" && value !== null) {
      for (const [key, child] of Object.entries(value)) {
        visit(child, prefix ? `${prefix}.${key}` : key);
      }
      return;
    }
    if (prefix) paths.add(prefix);
  };
  visit(data, "");
  return paths;
}

async function failureCopyPatterns(): Promise<RegExp[]> {
  const source = await readFile(
    resolve(repositoryRoot, "copy/en.yaml"),
    "utf8",
  );
  const patterns: RegExp[] = [];
  for (const line of source.split(/\r?\n/)) {
    const match = /^\s*([\w-]+):\s*(?:"([^"]*)"|'([^']*)')\s*(?:#.*)?$/.exec(
      line,
    );
    if (!match) continue;
    const key = match[1] ?? "";
    const value = (match[2] ?? match[3] ?? "").trim();
    if (
      /(?:amount_error|name_error|mapping_error|units_error|invalid|required|validation)/i.test(
        key,
      )
    )
      continue;
    const failureKey = /(?:error|failed|failure|could_not|couldnt)/i.test(key);
    if (!value || !failureKey) continue;
    const expression = value
      .split(/(\{[A-Za-z][A-Za-z0-9_]*\})/g)
      .map((part) =>
        /^\{[A-Za-z][A-Za-z0-9_]*\}$/.test(part)
          ? ".{0,100}?"
          : escapeRegExp(normalized(part)),
      )
      .join("");
    if (expression) patterns.push(new RegExp(expression, "iu"));
  }
  return patterns;
}

function appRouteTemplate(path: string): string {
  const pathname = path.split(/[?#]/)[0] || "/";
  return pathname
    .replace(/^\/money\/asset\/[^/]+$/, "/money/asset/:id")
    .replace(/^\/money\/holding\/[^/]+$/, "/money/holding/:id")
    .replace(/^\/money\/loan\/[^/]+$/, "/money/loan/:id")
    .replace(/^\/import\/[^/]+$/, "/import/:jobId")
    .replace(/^\/inbox\/brief\/[^/]+$/, "/inbox/brief/:week");
}

function slug(screen: string): string {
  return (
    screen
      .replace(/:id/g, "id")
      .replace(/:jobId/g, "job")
      .replace(/:week/g, "week")
      .replace(/^\/+|\/+$/g, "")
      .replace(/[^a-zA-Z0-9]+/g, "-")
      .toLowerCase() || "home"
  );
}

function isApiUrl(value: string, apiOrigin: string): boolean {
  try {
    const url = new URL(value);
    return (
      url.origin === apiOrigin &&
      (url.pathname === "/healthz" || url.pathname.startsWith("/v1/"))
    );
  } catch {
    return false;
  }
}

async function warmApi(apiUrl: string): Promise<void> {
  const healthUrl = new URL("/healthz", apiUrl).toString();
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(healthUrl, {
        cache: "no-store",
        signal: AbortSignal.timeout(10_000),
      });
      if (response.ok) return;
    } catch {
      // The free API can take time to wake up.
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 1_000));
  }
  throw new Error("The API did not become ready within 90 seconds.");
}

function screenInfo(): {
  text: string;
  jargonText: string;
  overflow: boolean;
  notes: number;
  alertTexts: string[];
  longSentences: number;
} {
  const visibleText: string[] = [];
  const jargonText: string[] = [];
  const isHoldingPage = window.location.pathname.startsWith("/money/holding/");
  const isIdeasPage = window.location.pathname.startsWith("/ideas");
  const visible = (element: Element): boolean => {
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return (
      style.visibility !== "hidden" &&
      style.display !== "none" &&
      rect.width > 0 &&
      rect.height > 0
    );
  };
  const isUserEnteredName = (parent: Element): boolean => {
    if (parent.closest("[data-security]")) return true;
    if (isIdeasPage) {
      if (parent.closest('[data-copy-screen="ideas"] li > p.mt-1.text-sm')) {
        return true;
      }
      const details = parent.closest('[data-copy-screen="ideas"] li details');
      const summary = details?.querySelector(":scope > summary")?.textContent;
      if (details && /broker/i.test(summary ?? "") && parent.closest("p")) {
        return true;
      }
    }
    if (isHoldingPage && parent.closest("main header h1, main header h2")) {
      return true;
    }
    if (isHoldingPage) {
      const detail = parent.closest("dd");
      const label = detail?.parentElement?.querySelector("dt")?.textContent;
      if (label?.trim().toLowerCase() === "code") return true;
    }
    return false;
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  while (node) {
    const parent = node.parentElement;
    if (
      parent &&
      !parent.closest("[data-term-copy]") &&
      !parent.closest("[hidden], [aria-hidden='true']")
    ) {
      const range = document.createRange();
      range.selectNode(node);
      if (range.getClientRects().length) {
        const value = node.textContent ?? "";
        visibleText.push(value);
        if (!isUserEnteredName(parent)) jargonText.push(value);
      }
    }
    node = walker.nextNode();
  }
  const text = visibleText.join(" ");
  const width = Math.round(window.visualViewport?.width ?? window.innerWidth);
  const warnings = Array.from(
    document.querySelectorAll(
      'main [class*="bg-warning-soft"], main [data-warning], main [data-highlight]',
    ),
  ).filter(visible);
  const highlights = Array.from(
    document.querySelectorAll('main section[aria-label="Helpful notes"] li'),
  ).filter(visible);
  const alerts = Array.from(document.querySelectorAll("[role=alert]")).filter(
    visible,
  );
  let longSentences = 0;
  for (const element of Array.from(document.querySelectorAll("p, li, dd"))) {
    if (!visible(element)) continue;
    const elementText = (element.textContent ?? "").replace(/\s+/g, " ").trim();
    if (!elementText) continue;
    const tokens =
      elementText.match(/[\p{L}\p{N}]+(?:['\u2019][\p{L}\p{N}]+)*/gu) ?? [];
    if (!tokens.length) continue;
    const nonLanguageTokens = tokens.filter((token) =>
      /^(?:[\d,.]+|inr|dkk|usd|eur|gbp|kr|rs)$/i.test(token),
    ).length;
    if (nonLanguageTokens / tokens.length > 0.5) continue;
    for (const sentence of elementText.match(/[^.!?]+[.!?]?/g) ?? []) {
      const words = sentence.match(/[\p{L}]+(?:['\u2019][\p{L}]+)*/gu) ?? [];
      if (words.length > 20) longSentences += 1;
    }
  }
  return {
    text,
    jargonText: jargonText.join(" "),
    overflow: window.innerWidth !== width,
    notes: new Set([...warnings, ...highlights]).size,
    alertTexts: alerts
      .map((element) => (element.textContent ?? "").replace(/\s+/g, " ").trim())
      .filter(Boolean),
    longSentences,
  };
}
function omitLotPhrase(text: string): string {
  return text.replace(/\ba lot(?: of)?\b/gi, " ");
}

async function countAndClearOfflineWrites(page: Page): Promise<number> {
  return page.evaluate(async () => {
    const databaseName = "wealth-money-map-offline";
    const storeName = "asset-writes";
    if (typeof indexedDB.databases === "function") {
      const databases = await indexedDB.databases();
      if (!databases.some((database) => database.name === databaseName))
        return 0;
    }
    const database = await new Promise<IDBDatabase>(
      (resolveDatabase, rejectDatabase) => {
        const request = indexedDB.open(databaseName, 1);
        request.onsuccess = () => resolveDatabase(request.result);
        request.onerror = () => rejectDatabase(request.error);
      },
    );
    if (!database.objectStoreNames.contains(storeName)) {
      database.close();
      return 0;
    }
    return new Promise<number>((resolveCount, rejectCount) => {
      const transaction = database.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      const countRequest = store.count();
      store.clear();
      countRequest.onerror = () => rejectCount(countRequest.error);
      transaction.oncomplete = () => {
        const count = countRequest.result;
        database.close();
        resolveCount(count);
      };
      transaction.onerror = () => rejectCount(transaction.error);
      transaction.onabort = () => rejectCount(transaction.error);
    });
  });
}

function reportLines(findings: Finding[], severity: Severity): string[] {
  const groups = new Map<string, Finding[]>();
  for (const finding of findings.filter((item) => item.severity === severity)) {
    const key = JSON.stringify([finding.screen, finding.check, finding.detail]);
    groups.set(key, [...(groups.get(key) ?? []), finding]);
  }
  if (!groups.size) return ["No findings."];
  const lines: string[] = [];
  const ordered = [...groups.values()].sort((left, right) => {
    const a = left[0];
    const b = right[0];
    return (
      (a?.screen ?? "").localeCompare(b?.screen ?? "") ||
      (a?.check ?? "").localeCompare(b?.check ?? "") ||
      (a?.detail ?? "").localeCompare(b?.detail ?? "")
    );
  });
  let previousScreen = "";
  for (const items of ordered) {
    const first = items[0];
    if (!first) continue;
    if (first.screen !== previousScreen) {
      if (previousScreen) lines.push("");
      lines.push("### " + first.screen, "");
      previousScreen = first.screen;
    }
    lines.push(
      "- **" +
        first.check +
        ":** " +
        first.detail +
        " ×" +
        items.length +
        " · first screenshot: `" +
        first.screenshot +
        "`",
    );
  }
  lines.push("");
  return lines;
}

export const test = base.extend<{ live: LiveRuntime }>({
  live: [
    async ({}, use) => {
      const environment = getLiveEnvironment() as LiveEnvironment;
      const apiOrigin = new URL(environment.apiUrl).origin;
      const screensDir = resolve(environment.outDir, "screens");
      await mkdir(screensDir, { recursive: true });
      const copyPatterns = await failureCopyPatterns();
      const copyPaths = await copyKeyPaths();
      const bannedTermList = await bannedTerms();
      const findings: Finding[] = [];
      const snapshots = new Map<string, unknown>();
      const coverage = new Map<string, Coverage>();
      const requestStarted = new Map<Request, number>();
      const responseTasks = new Set<Promise<void>>();
      const blockedRequests = new WeakSet<Request>();
      let currentScreen = "/";
      let screenshotNumber = 0;
      let activeScreenshot: string | null = null;
      let latestScreenshot: string | null = null;
      const screenshotName = (number: number, screen: string) =>
        String(number).padStart(3, "0") + "-" + slug(screen) + ".png";
      let context: BrowserContext;

      try {
        context = await chromium.launchPersistentContext(
          environment.profileDir,
          {
            ...(environment.chromePath
              ? { executablePath: environment.chromePath }
              : { channel: "chrome" as const }),
            headless: !environment.headed,
            viewport: { width: 360, height: 740 },
            deviceScaleFactor: 1,
            isMobile: true,
            hasTouch: true,
            serviceWorkers: "block",
          },
        );
      } catch (error) {
        const underlying = safeOutput(
          firstLine(error instanceof Error ? error.message : String(error)),
          environment,
        ).slice(0, 200);
        throw new Error(
          "Could not open the browser profile. If the sign-in window is still open, close it first. " +
            "Underlying error: " +
            underlying,
        );
      }

      const page = context.pages()[0] ?? (await context.newPage());
      const addFinding = (
        severity: Severity,
        check: string,
        detail: string,
      ) => {
        findings.push({
          severity,
          screen: currentScreen,
          check,
          detail: safeOutput(detail, environment),
          screenshot:
            activeScreenshot ??
            latestScreenshot ??
            screenshotName(screenshotNumber + 1, currentScreen),
        });
      };
      const recordResponse = async (response: Response) => {
        const request = response.request();
        const responseUrl = response.url();
        if (!isApiUrl(responseUrl, apiOrigin) || blockedRequests.has(request))
          return;
        const method = request.method().toUpperCase();
        const path = routeTemplate(responseUrl, {
          pwaUrl: environment.pwaUrl,
          apiUrl: environment.apiUrl,
        });
        if (
          response.status() >= 500 ||
          (method === "GET" && response.status() >= 400)
        ) {
          addFinding(
            "fail",
            "API response",
            method + " " + path + " returned status " + response.status() + ".",
          );
        }
        const started = requestStarted.get(request);
        if (started !== undefined && Date.now() - started > 5_000) {
          addFinding(
            "note",
            "API response time",
            method + " " + path + " exceeded 5 seconds.",
          );
        }
        if (method === "GET" && response.ok()) {
          try {
            const url = new URL(responseUrl);
            snapshots.set(url.pathname, await response.json());
          } catch {
            // Non-JSON reads are not needed for list cross-checks.
          }
        }
        requestStarted.delete(request);
      };
      const attachPage = (observedPage: Page) => {
        observedPage.on("request", (request) =>
          requestStarted.set(request, Date.now()),
        );
        observedPage.on("response", (response) => {
          let task: Promise<void>;
          task = recordResponse(response)
            .catch(() => {
              addFinding(
                "fail",
                "API response",
                "An API response could not be inspected.",
              );
            })
            .finally(() => responseTasks.delete(task));
          responseTasks.add(task);
        });
        observedPage.on("pageerror", (error) =>
          addFinding(
            "fail",
            "page error",
            safeOutput(firstLine(error.message), environment).slice(0, 200),
          ),
        );
        observedPage.on("console", (message) => {
          if (message.type() === "error") {
            addFinding(
              "fail",
              "console error",
              safeOutput(message.text(), environment).slice(0, 200),
            );
          }
        });
      };
      attachPage(page);
      context.on("page", (newPage) => {
        if (newPage !== page) attachPage(newPage);
      });

      await context.route("**/*", async (route: Route) => {
        const request = route.request();
        const requestUrl = request.url();
        const isNavigation = request.isNavigationRequest();
        let frameIsMain = false;
        try {
          const frame = request.frame();
          frameIsMain = frame === frame.page().mainFrame();
        } catch {
          // Requests without a page frame cannot be main-frame navigations.
        }
        const method = request.method().toUpperCase();
        const decision = decideRequest(
          { method, url: requestUrl, isNavigation, frameIsMain },
          { pwaUrl: environment.pwaUrl, apiUrl: environment.apiUrl },
        );
        if (decision === "allow") {
          await route.continue();
          return;
        }
        blockedRequests.add(request);
        addFinding(
          "fail",
          "blocked request",
          method +
            " " +
            routeTemplate(requestUrl, {
              pwaUrl: environment.pwaUrl,
              apiUrl: environment.apiUrl,
            }) +
            " was blocked.",
        );
        let isApiWrite = false;
        try {
          isApiWrite =
            apiWrites.has(method) && new URL(requestUrl).origin === apiOrigin;
        } catch {
          isApiWrite = false;
        }
        if (isApiWrite) {
          await route.fulfill({
            status: 403,
            contentType: "application/json",
            headers: {
              "access-control-allow-origin": new URL(environment.pwaUrl).origin,
              "access-control-allow-credentials": "true",
            },
            body: JSON.stringify({ detail: "blocked by live look-only check" }),
          });
        } else {
          await route.abort("blockedbyclient");
        }
      });

      await warmApi(environment.apiUrl);
      const runtime: LiveRuntime = {
        page,
        environment,
        findings,
        snapshots,
        setScreen(path) {
          currentScreen = appRouteTemplate(path);
          if (!coverage.has(currentScreen)) {
            coverage.set(currentScreen, {
              screen: currentScreen,
              expanders: 0,
              dialogs: 0,
              listItems: 0,
            });
          }
        },
        addFinding,
        count(field, amount = 1) {
          const item = coverage.get(currentScreen) ?? {
            screen: currentScreen,
            expanders: 0,
            dialogs: 0,
            listItems: 0,
          };
          item[field] += amount;
          coverage.set(currentScreen, item);
        },
        async screenshot() {
          const filename =
            activeScreenshot ??
            screenshotName(screenshotNumber + 1, currentScreen);
          screenshotNumber += 1;
          try {
            await page.screenshot({
              path: resolve(screensDir, filename),
              fullPage: true,
              animations: "disabled",
              caret: "hide",
            });
          } catch {
            addFinding(
              "fail",
              "screenshot",
              "A screen screenshot could not be saved.",
            );
          }
          latestScreenshot = filename;
          activeScreenshot = null;
        },
        async checkScreen(stateStartedAt = Date.now()) {
          activeScreenshot = screenshotName(
            screenshotNumber + 1,
            currentScreen,
          );
          try {
            await page.waitForLoadState("networkidle", { timeout: 8_000 });
          } catch {
            // Record a slow screen and continue collecting findings.
          }
          // Loading placeholders pulse until their data arrives; checking earlier
          // would review an empty skeleton instead of the screen.
          try {
            await page.waitForFunction(
              () =>
                !document.querySelector(
                  "main .animate-pulse, main [aria-busy='true']",
                ),
              undefined,
              { timeout: 30_000 },
            );
          } catch {
            addFinding(
              "note",
              "still loading",
              "Loading placeholders were still visible after 30 seconds.",
            );
          }
          while (responseTasks.size) {
            await Promise.all([...responseTasks]);
          }
          if (Date.now() - stateStartedAt > 8_000) {
            addFinding(
              "note",
              "screen settle time",
              "The screen took over 8 seconds to settle.",
            );
          }
          let info: ReturnType<typeof screenInfo>;
          try {
            info = await page.evaluate(screenInfo);
          } catch {
            addFinding(
              "fail",
              "screen check",
              "The visible screen could not be checked.",
            );
            await this.screenshot();
            return;
          }
          const text = normalized(info.text);
          if (!text.trim()) {
            addFinding(
              "fail",
              "screen content",
              "No visible screen content was found.",
            );
          }
          const brokenText = [
            {
              check: "broken text",
              pattern: /\b(?:nan|undefined|null|infinity)\b/i,
            },
            { check: "broken text", pattern: /\[object object\]/i },
            {
              check: "unreplaced placeholder",
              pattern: /\{[a-z][a-z0-9_]*\}/i,
            },
          ];
          for (const item of brokenText) {
            const match = item.pattern.exec(text);
            if (match?.index !== undefined) {
              addFinding(
                "fail",
                item.check,
                safeOutput(
                  evidenceDetail(text, match.index, match[0].length),
                  environment,
                ).slice(0, 120),
              );
            }
          }
          const copyKeyPattern =
            /[\p{L}_][\p{L}\p{N}_-]*(?:\.[\p{L}_][\p{L}\p{N}_-]*)*/gu;
          for (const match of info.text.matchAll(copyKeyPattern)) {
            if (copyPaths.has(match[0]) && match.index !== undefined) {
              addFinding(
                "fail",
                "raw copy key",
                safeOutput(
                  evidenceDetail(info.text, match.index, match[0].length),
                  environment,
                ).slice(0, 120),
              );
            }
          }
          if (info.overflow) {
            addFinding(
              "fail",
              "horizontal overflow",
              "The page is wider than the 360-pixel screen.",
            );
          }
          for (const pattern of copyPatterns) {
            const match = pattern.exec(text);
            if (match?.index !== undefined) {
              addFinding(
                "fail",
                "error copy",
                safeOutput(
                  evidenceDetail(text, match.index, match[0].length),
                  environment,
                ).slice(0, 120),
              );
            }
          }
          for (const alertText of info.alertTexts) {
            addFinding(
              "note",
              "alert",
              safeOutput(alertText, environment).slice(0, 120),
            );
          }
          const languageText = omitLotPhrase(normalized(info.jargonText));
          for (const bannedTerm of bannedTermList) {
            if (
              bannedTerm.allowed_context === "import_statement" &&
              currentScreen.startsWith("/import")
            ) {
              continue;
            }
            const pattern = new RegExp(
              "\\b" + escapeRegExp(bannedTerm.term) + "\\b",
              "iu",
            );
            const match = pattern.exec(languageText);
            if (match?.index !== undefined) {
              addFinding(
                "note",
                "jargon",
                safeOutput(
                  evidenceDetail(
                    languageText,
                    match.index,
                    match[0].length,
                    `Jargon "${bannedTerm.term}"`,
                  ),
                  environment,
                ).slice(0, 120),
              );
            }
          }
          if (info.longSentences) {
            addFinding(
              "note",
              "sentence length",
              info.longSentences + " visible sentence(s) exceed 20 words.",
            );
          }
          const noteCount = info.notes + info.alertTexts.length;
          if (noteCount > 3) {
            addFinding(
              "note",
              "notes and warnings",
              noteCount + " note or warning item(s) are visible.",
            );
          }
          try {
            const result = await new AxeBuilder({ page }).analyze();
            for (const violation of result.violations) {
              if (
                violation.impact === "serious" ||
                violation.impact === "critical"
              ) {
                addFinding(
                  "note",
                  "accessibility",
                  "Rule " +
                    violation.id +
                    ": " +
                    violation.nodes.length +
                    " node(s); first target: " +
                    (violation.nodes[0]?.target?.join(" >>> ") ??
                      "unavailable"),
                );
              }
            }
          } catch {
            addFinding(
              "fail",
              "accessibility scan",
              "Axe could not scan the screen.",
            );
          }
          await this.screenshot();
        },
        async writeReports() {
          const ordered = [...findings].sort(
            (a, b) =>
              a.severity.localeCompare(b.severity) ||
              a.screen.localeCompare(b.screen),
          );
          const fails = findings.filter(
            (item) => item.severity === "fail",
          ).length;
          const notes = findings.filter(
            (item) => item.severity === "note",
          ).length;
          await mkdir(environment.outDir, { recursive: true });
          await writeFile(
            resolve(environment.outDir, "findings.json"),
            JSON.stringify(ordered, null, 2) + "\n",
            "utf8",
          );
          const lines = [
            "# Live look-only findings",
            "",
            "Fail findings: " + fails,
            "Note findings: " + notes,
            "",
            "## Fail findings",
            "",
            ...reportLines(ordered, "fail"),
            "## Note findings",
            "",
            ...reportLines(ordered, "note"),
            "## Coverage",
            "",
            "| Screen route | Expanders | Dialogs | List items followed |",
            "| --- | ---: | ---: | ---: |",
            ...[...coverage.values()]
              .sort((a, b) => a.screen.localeCompare(b.screen))
              .map(
                (item) =>
                  "| " +
                  item.screen +
                  " | " +
                  item.expanders +
                  " | " +
                  item.dialogs +
                  " | " +
                  item.listItems +
                  " |",
              ),
            "",
          ];
          await writeFile(
            resolve(environment.outDir, "findings.md"),
            lines.join("\n"),
            "utf8",
          );
          return { fails, notes };
        },
      };

      try {
        await use(runtime);
      } finally {
        try {
          const queued = await countAndClearOfflineWrites(page);
          if (queued) {
            addFinding(
              "fail",
              "offline queue",
              queued + " queued write(s) were found and cleared.",
            );
          }
        } catch {
          addFinding(
            "fail",
            "offline queue",
            "The offline queue could not be inspected and cleared.",
          );
        }
        await runtime.writeReports();
        await context.close();
      }
    },
    { scope: "worker" },
  ],
});
