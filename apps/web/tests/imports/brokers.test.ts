import { describe, expect, it, vi } from "vitest";
import {
  BROKER_GROUPS,
  BROKER_STORAGE_PREFIX,
  defaultBrokerForResidence,
  readBrokerForMember,
  saveBrokerForMember,
} from "../../src/imports/brokers";

describe("broker choices", () => {
  it("groups the supported brokers with plain labels", () => {
    expect(BROKER_GROUPS.map(({ label }) => label)).toEqual([
      "India",
      "US shares from India",
      "Denmark",
      "Other",
    ]);
    expect(
      BROKER_GROUPS.flatMap((group) => group.options.map(([value]) => value)),
    ).toContain("hdfc_securities");
    expect(
      BROKER_GROUPS.flatMap((group) => group.options.map(([value]) => value)),
    ).toContain("other_danish_broker");
  });

  it("uses a residence default and remembers a choice separately per member", () => {
    localStorage.clear();
    expect(defaultBrokerForResidence("IN")).toBe("zerodha");
    expect(defaultBrokerForResidence("DK")).toBe("nordnet");
    expect(defaultBrokerForResidence("GB")).toBe("other");

    saveBrokerForMember("member-in", "hdfc_securities");
    saveBrokerForMember("member-dk", "saxo");
    expect(readBrokerForMember("member-in", "IN")).toBe("hdfc_securities");
    expect(readBrokerForMember("member-dk", "DK")).toBe("saxo");
    expect(readBrokerForMember("member-new", "IN")).toBe("zerodha");
    expect(localStorage.getItem(`${BROKER_STORAGE_PREFIX}member-in`)).toBe(
      "hdfc_securities",
    );
  });

  it("keeps usable defaults when local storage throws", () => {
    const getItem = vi
      .spyOn(Storage.prototype, "getItem")
      .mockImplementation(() => {
        throw new Error("storage is disabled");
      });
    const setItem = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("storage is disabled");
      });

    expect(readBrokerForMember("member-in", "IN")).toBe("zerodha");
    expect(() => saveBrokerForMember("member-in", "nordnet")).not.toThrow();

    getItem.mockRestore();
    setItem.mockRestore();
  });
});
