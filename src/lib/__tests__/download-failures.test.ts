import { describe, expect, it } from "vitest";
import {
  buildDownloadFailureMessage,
  toConvertError,
} from "../download-failures";
import type { FailureKind, FailureSummary } from "../types";

function summary(...groups: Array<[FailureKind, number, string[]]>): FailureSummary {
  return {
    groups: groups.map(([kind, count, examples]) => ({
      kind,
      count,
      examples,
      http_statuses:
        kind === "access_denied"
          ? [403]
          : kind === "not_found"
            ? [404]
            : [],
    })),
    expiry: null,
  };
}

describe("buildDownloadFailureMessage", () => {
  it("returns no message for a complete conversion", () => {
    expect(buildDownloadFailureMessage(summary(), 0, false)).toBeNull();
  });

  it("explains an all-expired hard failure", () => {
    expect(
      buildDownloadFailureMessage(
        summary(["expired_url", 2, ["one.jpg", "two.jpg"]]),
        2,
        true,
      ),
    ).toMatchObject({
      primary:
        "Your download links have expired. Export the dataset again from Ultralytics Platform, then retry.",
      breakdown: [
        "2 download links expired — one.jpg, two.jpg. Export the dataset again.",
      ],
    });
  });

  it("prioritizes confirmed all-expired links over missing-link failures", () => {
    const failureSummary = summary(
      ["missing_url", 1, ["missing.jpg"]],
      ["expired_url", 1, ["expired.jpg"]],
    );
    failureSummary.expiry = {
      urls_with_expiry: 1,
      expired_urls: 1,
      all_expired: true,
      latest_expired_at: 1_784_419_200,
    };

    const message = buildDownloadFailureMessage(
      failureSummary,
      2,
      true,
    );

    expect(message?.primary).toBe(
      "Your download links expired on 19 Jul 2026. Export the dataset again from Ultralytics Platform, then retry.",
    );
    expect(message?.breakdown).toContain(
      "If this export is new, check that your computer's date and time are correct.",
    );
    expect(message?.diagnostics).toContain(
      "latest_expired_at: 1784419200",
    );
  });

  it("explains 403 without claiming expiry", () => {
    expect(
      buildDownloadFailureMessage(
        summary(["access_denied", 3, ["one.jpg"]]),
        3,
        true,
      )?.primary,
    ).toBe(
      "3 images could not be downloaded because their links expired or access was denied. Export the dataset again.",
    );
  });

  it("combines connect and DNS failures into one remedy", () => {
    expect(
      buildDownloadFailureMessage(
        summary(
          ["connect", 2, ["one.jpg"]],
          ["dns", 1, ["two.jpg"]],
        ),
        3,
        true,
      )?.primary,
    ).toBe(
      "Couldn't connect to the image server. Check your internet connection and try again.",
    );
  });

  it("honestly explains blocked local addresses", () => {
    expect(
      buildDownloadFailureMessage(
        summary(["blocked_address", 1, ["one.jpg"]]),
        1,
        true,
      )?.primary,
    ).toBe(
      "On this network, the image server points to a local address, so the app stopped the download for safety. Try a different network, or ask your network administrator.",
    );
  });

  it("gives plain remedies for not found, timeout, server, and oversized failures", () => {
    expect(
      buildDownloadFailureMessage(
        summary(["not_found", 2, []]),
        2,
        true,
      )?.primary,
    ).toBe(
      "2 images could not be found on the server. Export the dataset again. If it keeps happening, those images may have been deleted from the dataset.",
    );
    expect(
      buildDownloadFailureMessage(summary(["timeout", 2, []]), 2, true)
        ?.primary,
    ).toBe(
      "2 downloads timed out. Your connection may be slow. Try again.",
    );
    expect(
      buildDownloadFailureMessage(
        summary(["server_error", 2, []]),
        2,
        true,
      )?.primary,
    ).toBe(
      "The image server had a problem. Wait a moment and try again.",
    );
    expect(
      buildDownloadFailureMessage(summary(["too_large", 2, []]), 2, false)
        ?.primary,
    ).toBe(
      "2 images and their corresponding annotations were skipped. 2 images were larger than 50 MB. Use smaller images and try again.",
    );
  });

  it.each([
    [
      "malformed_url" as const,
      "2 download links were invalid. Export the dataset again and retry.",
    ],
    [
      "unsupported_scheme" as const,
      "2 download links used an unsupported format. Export the dataset again and retry.",
    ],
    [
      "response_error" as const,
      "2 downloads stopped before finishing. Check your connection and try again.",
    ],
    [
      "size_overflow" as const,
      "2 images reported an invalid size. Export the dataset again and retry.",
    ],
    [
      "http_error" as const,
      "The image server rejected 2 downloads. Export the dataset again or try later.",
    ],
    [
      "download_error" as const,
      "2 downloads failed. Check your internet connection and try again.",
    ],
  ])("provides a remedy for %s", (kind, expected) => {
    expect(
      buildDownloadFailureMessage(summary([kind, 2, []]), 2, true)?.primary,
    ).toBe(expected);
  });

  it("uses generic mixed copy and only references existing details", () => {
    const message = buildDownloadFailureMessage(
      summary(
        ["not_found", 2, ["one.jpg"]],
        ["timeout", 2, ["two.jpg"]],
      ),
      4,
      false,
    );

    expect(message?.primary).toBe(
      "4 images and their corresponding annotations were skipped. See details below for causes and next steps.",
    );
    expect(message?.breakdown).toHaveLength(2);
    expect(message?.diagnostics).toContain("not_found: 2");
    expect(message?.diagnostics).not.toContain("https://");
  });

  it("does not attribute every mixed failure to the dominant cause", () => {
    expect(
      buildDownloadFailureMessage(
        summary(
          ["expired_url", 2, ["one.jpg", "two.jpg"]],
          ["not_found", 1, ["three.jpg"]],
        ),
        3,
        false,
      )?.primary,
    ).toBe(
      "3 images and their corresponding annotations were skipped. See details below for causes and next steps.",
    );
  });
});

describe("toConvertError", () => {
  it("preserves a structured Tauri error", () => {
    const failureSummary = summary(["expired_url", 1, ["one.jpg"]]);
    const error = toConvertError({
      kind: "download_failed",
      message: "Images could not be downloaded.",
      failure_summary: failureSummary,
    });

    expect(error.failure_summary).toEqual(failureSummary);
    expect(error.kind).toBe("download_failed");
  });

  it("falls back safely for ordinary and malformed errors", () => {
    expect(toConvertError(new Error("Disk full")).message).toBe("Disk full");
    expect(toConvertError("Bad input").message).toBe("Bad input");
    expect(toConvertError({ secret: "value" })).toEqual({
      kind: "conversion_failed",
      message: "Conversion failed.",
      failure_summary: null,
    });
  });
});
