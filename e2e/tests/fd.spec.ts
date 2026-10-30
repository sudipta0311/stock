import {
  expect,
  openFixedDepositForm,
  test,
  completeFixedDeposit,
  writeReport,
} from "../support/app";
import { startReadyAccount } from "../support/app";

test("fixed deposit saves its estimate within thirty seconds", async ({
  page,
}, testInfo) => {
  await startReadyAccount(page, "fixed-deposit");
  await openFixedDepositForm(page);
  const elapsedMs = await completeFixedDeposit(page);
  expect(elapsedMs).toBeLessThan(30_000);
  await writeReport("fixed-deposit", testInfo.project.name, {
    elapsedMs,
    targetMs: 30_000,
  });
});
