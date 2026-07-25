import { describe, expect, it } from "vitest";
import { getDownloadWarning } from "../converter-screen";
import type { ConvertResult } from "@/lib/types";

const completeResult: ConvertResult = {
  zip_path: "/tmp/dataset.zip",
  file_count: 4,
  image_count: 2,
  download_total: 2,
  failed_downloads: 0,
  omitted_images: 0,
  expired_url_failures: 0,
};

describe("getDownloadWarning", () => {
  it("returns no warning when no records were omitted", () => {
    expect(getDownloadWarning(completeResult)).toBeNull();
  });

  it("reports omitted images and annotations including missing URLs", () => {
    expect(
      getDownloadWarning({
        ...completeResult,
        download_total: 1,
        failed_downloads: 0,
        omitted_images: 1,
      }),
    ).toBe("1 image and its corresponding annotations were omitted.");
  });

  it("adds an expired signed URL hint for 403 responses", () => {
    expect(
      getDownloadWarning({
        ...completeResult,
        download_total: 3,
        failed_downloads: 2,
        omitted_images: 2,
        expired_url_failures: 2,
      }),
    ).toBe(
      "2 images and their corresponding annotations were omitted. 2 downloads returned HTTP 403; signed URLs may have expired. Re-export the dataset and try again.",
    );
  });
});
