import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { runInNewContext } from "node:vm";
import { describe, expect, it, vi } from "vitest";

const source = readFileSync(
  resolve(process.cwd(), "../../tools/gmail-statements/Code.gs"),
  "utf8",
)
  .replace(
    /const API_URL = "[^"]*";/,
    'const API_URL = "https://api.example.test/v1/statement-inbox/uploads";',
  )
  .replace(
    /const UPLOAD_KEY = "[^"]*";/,
    'const UPLOAD_KEY = "wsk_synthetic-setup-key";',
  );

interface FakeResponse {
  getResponseCode(): number;
  getContentText(): string;
}

function response(status: number, body = ""): FakeResponse {
  return {
    getResponseCode: () => status,
    getContentText: () => body,
  };
}

function makeHarness(
  fetchImpl: () => FakeResponse,
  senders: string[] = ["statements@camsonline.com"],
) {
  const searches: string[] = [];
  const logs: string[] = [];
  const posts: Array<{ url: string; options: Record<string, unknown> }> = [];
  const labels: string[] = [];
  const sleeps: number[] = [];
  const createdLabels: string[] = [];
  const triggers: Array<{ handler: string; days?: number; hour?: number }> = [];
  const properties = new Map<string, string>();
  properties.set(
    "API_URL",
    "https://api.example.test/v1/statement-inbox/uploads",
  );
  properties.set("UPLOAD_KEY", "wsk_synthetic-test-key");
  const attachment = {
    getContentType: () => "application/pdf",
    getName: () => "statement.pdf",
    setName: vi.fn(function (this: { getName: () => string }) {
      return this;
    }),
  };
  const threads = senders.map((from, index) => ({
    getMessages: () => [
      {
        getAttachments: () => [attachment],
        getFrom: () => from,
        getSubject: () => "Monthly statement",
        getDate: () => new Date("2026-10-01T07:00:00Z"),
        getId: () => `synthetic-message-${index}`,
      },
    ],
    addLabel: vi.fn((label: { name: string }) => labels.push(label.name)),
  }));
  const thread = threads[0];
  const gmailLabel = { name: "Wealth/Imported" };
  const scriptProperties = {
    setProperties: (values: Record<string, string>) => {
      Object.entries(values).forEach(([key, value]) =>
        properties.set(key, value),
      );
    },
    setProperty: (key: string, value: string) => properties.set(key, value),
    getProperty: (key: string) => properties.get(key) ?? null,
  };
  const context = {
    GmailApp: {
      search: (query: string) => {
        searches.push(query);
        return threads;
      },
      getUserLabelByName: () => gmailLabel,
      createLabel: (name: string) => {
        createdLabels.push(name);
        return { name };
      },
    },
    UrlFetchApp: {
      fetch: (url: string, options: Record<string, unknown>) => {
        posts.push({ url, options });
        return fetchImpl();
      },
    },
    PropertiesService: { getScriptProperties: () => scriptProperties },
    console: { log: (text: string) => logs.push(text) },
    Utilities: { sleep: (delay: number) => sleeps.push(delay) },
    ScriptApp: {
      getProjectTriggers: () => [],
      newTrigger: (handler: string) => {
        const trigger: { handler: string; days?: number; hour?: number } = {
          handler,
        };
        triggers.push(trigger);
        return {
          timeBased: () => ({
            everyDays: (days: number) => {
              trigger.days = days;
              return {
                atHour: (hour: number) => {
                  trigger.hour = hour;
                  return { create: () => trigger };
                },
              };
            },
          }),
        };
      },
      deleteTrigger: vi.fn(),
    },
    module: { exports: {} as Record<string, unknown> },
  };
  runInNewContext(
    `${source}\nmodule.exports = { setup, run, searchQuery: _searchQuery };`,
    context,
  );
  const exported = context.module.exports as {
    setup: () => { sent: number; labelled: number };
    run: () => { sent: number; labelled: number };
    searchQuery: () => string;
  };
  return {
    exported,
    searches,
    logs,
    posts,
    labels,
    sleeps,
    createdLabels,
    triggers,
    properties,
    thread,
  };
}

describe("Gmail statement helper", () => {
  it("sets up a label and a daily morning trigger, then checks Gmail once now", () => {
    const harness = makeHarness(() => response(202));

    expect(harness.exported.setup()).toEqual({ sent: 1, labelled: 1 });

    expect(harness.properties.get("API_URL")).toBe(
      "https://api.example.test/v1/statement-inbox/uploads",
    );
    expect(harness.properties.get("UPLOAD_KEY")).toBe(
      "wsk_synthetic-setup-key",
    );
    expect(harness.triggers).toEqual([{ handler: "run", days: 1, hour: 7 }]);
  });

  it("searches known senders, posts each PDF, then labels a successful thread", () => {
    const harness = makeHarness(() => response(202));

    const result = harness.exported.run();

    expect(result).toEqual({ sent: 1, labelled: 1 });
    expect(harness.searches.at(0)).toContain("has:attachment filename:pdf");
    expect(harness.searches.at(0)).toContain("newer_than:365d");
    expect(harness.searches.at(0)).toContain("-label:wealth-imported");
    for (const domain of [
      "camsonline.com",
      "kfintech.com",
      "mfcentral.com",
      "nsdl.co.in",
      "cdslstatement.com",
    ]) {
      expect(harness.searches.at(0)).toContain(`from:${domain}`);
    }
    expect(harness.posts).toHaveLength(1);
    expect(harness.posts.at(0)?.url).toBe(
      "https://api.example.test/v1/statement-inbox/uploads",
    );
    expect(harness.posts.at(0)?.options.method).toBe("post");
    expect(
      (harness.posts.at(0)?.options.headers as Record<string, string>)
        .Authorization,
    ).toBe("StatementKey wsk_synthetic-setup-key");
    expect(harness.labels).toEqual(["Wealth/Imported"]);
    expect(harness.logs).toEqual([
      "Found 1 statement email(s) from the last 365 days.",
      "Sent 1 statement PDF(s). Open Inbox in Wealth to enter the password.",
    ]);
  });

  it("sends only the newest statement per provider on the first run, then looks at recent mail", () => {
    const harness = makeHarness(
      () => response(202),
      [
        "donotreply@camsonline.com",
        "donotreply@camsonline.com",
        "ecas@cdslstatement.com",
      ],
    );

    expect(harness.exported.run()).toEqual({ sent: 2, labelled: 2 });
    expect(harness.posts).toHaveLength(2);

    harness.exported.run();
    expect(harness.searches.at(1)).toContain("newer_than:45d");
    expect(harness.posts).toHaveLength(5);
  });

  it("says so when there are no statement emails", () => {
    const harness = makeHarness(() => response(202), []);

    expect(harness.exported.run()).toEqual({ sent: 0, labelled: 0 });
    expect(harness.logs).toEqual([
      "No new statement emails from CAMS, KFintech, MF Central, NSDL or CDSL in the last 365 days.",
    ]);
  });

  it("explains a rejected upload key and keeps the first run for next time", () => {
    const harness = makeHarness(() => response(401));

    harness.exported.run();

    expect(harness.logs).toContain(
      "Wealth didn't accept the upload key. Create a new key in Settings and paste the new script.",
    );
    expect(harness.properties.get("FIRST_RUN_DONE")).toBeUndefined();
  });

  it("does not label a thread after a failed upload", () => {
    const harness = makeHarness(() => response(401));

    expect(harness.exported.run()).toEqual({ sent: 0, labelled: 0 });
    expect(harness.labels).toEqual([]);
  });

  it("waits and retries once after a server error", () => {
    let attempts = 0;
    const harness = makeHarness(() => {
      attempts += 1;
      return attempts === 1 ? response(503) : response(202);
    });

    expect(harness.exported.run()).toEqual({ sent: 1, labelled: 1 });
    expect(attempts).toBe(2);
    expect(harness.sleeps).toEqual([50000]);
    expect(harness.labels).toEqual(["Wealth/Imported"]);
  });

  it("labels an already received duplicate", () => {
    const harness = makeHarness(() => response(200, '{"status":"duplicate"}'));

    expect(harness.exported.run()).toEqual({ sent: 1, labelled: 1 });
    expect(harness.labels).toEqual(["Wealth/Imported"]);
  });
});
