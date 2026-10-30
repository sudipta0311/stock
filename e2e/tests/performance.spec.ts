import {
  expect,
  gotoApp,
  skipGuidedTour,
  startReadyAccount,
  test,
  writeReport,
} from "../support/app";

test("authenticated home meets the mobile LCP budget under CPU and network throttling @performance", async ({
  page,
}, testInfo) => {
  test.skip(
    testInfo.project.name !== "mobile",
    "Performance runs once on the 412 by 915 phone profile.",
  );
  await startReadyAccount(page, "performance");
  await gotoApp(page, "/");
  await skipGuidedTour(page);

  const session = await page.context().newCDPSession(page);
  await session.send("Network.enable");
  await session.send("Network.setCacheDisabled", { cacheDisabled: true });
  await session.send("Network.emulateNetworkConditions", {
    offline: false,
    latency: 150,
    downloadThroughput: 1_600_000 / 8,
    uploadThroughput: 750_000 / 8,
    connectionType: "cellular4g",
  });
  await session.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  await page.addInitScript(() => {
    const root = window as Window & { __e2eLcp?: number };
    root.__e2eLcp = undefined;
    new PerformanceObserver((entries) => {
      const latest = entries.getEntries().at(-1);
      if (latest) root.__e2eLcp = latest.startTime;
    }).observe({ type: "largest-contentful-paint", buffered: true });
  });

  const origin = process.env.E2E_BASE_URL ?? "";
  const requestPaths = new Map<string, string>();
  const scriptSizes = new Map<string, number>();
  session.on("Network.responseReceived", (event) => {
    if (event.type === "Script" && event.response.url.startsWith(origin)) {
      requestPaths.set(event.requestId, new URL(event.response.url).pathname);
    }
  });
  session.on("Network.loadingFinished", (event) => {
    const path = requestPaths.get(event.requestId);
    if (path)
      scriptSizes.set(
        path,
        (scriptSizes.get(path) ?? 0) + event.encodedDataLength,
      );
  });

  await page.goto(origin + "/", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "This week" })).toBeVisible();
  await page.waitForTimeout(750);
  const lcpMs = await page.evaluate(
    () => (window as Window & { __e2eLcp?: number }).__e2eLcp ?? null,
  );
  const scriptChunks = [...scriptSizes]
    .map(([path, bytes]) => ({ path, bytes }))
    .sort((left, right) => right.bytes - left.bytes);
  const totalTransferredJsBytes = scriptChunks.reduce(
    (total, chunk) => total + chunk.bytes,
    0,
  );
  await session.detach();

  await writeReport("perf", "all", {
    viewport: { width: 412, height: 915 },
    cpuThrottlingRate: 4,
    network: { latencyMs: 150, downloadBitsPerSecond: 1_600_000 },
    lcpMs,
    lcpTargetMs: 2_500,
    totalTransferredJsBytes,
    scriptChunks,
  });
  if (typeof lcpMs !== "number")
    throw new Error("The page did not report LCP.");
  expect(lcpMs).toBeLessThan(2_500);
});
