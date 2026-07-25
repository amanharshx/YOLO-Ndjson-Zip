import type {
  ConvertError,
  FailureGroup,
  FailureKind,
  FailureSummary,
} from "./types";

export interface DownloadFailureMessage {
  primary: string;
  breakdown: string[];
  diagnostics: string;
}

const FAILURE_KINDS = new Set<FailureKind>([
  "missing_url",
  "expired_url",
  "access_denied",
  "not_found",
  "timeout",
  "connect",
  "dns",
  "blocked_address",
  "malformed_url",
  "unsupported_scheme",
  "server_error",
  "response_error",
  "too_large",
  "size_overflow",
  "http_error",
  "download_error",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isFailureGroup(value: unknown): value is FailureGroup {
  return (
    isRecord(value) &&
    typeof value.kind === "string" &&
    FAILURE_KINDS.has(value.kind as FailureKind) &&
    typeof value.count === "number" &&
    Number.isSafeInteger(value.count) &&
    value.count >= 0 &&
    Array.isArray(value.examples) &&
    value.examples.every((example) => typeof example === "string") &&
    Array.isArray(value.http_statuses) &&
    value.http_statuses.every(
      (status) => typeof status === "number" && Number.isSafeInteger(status),
    )
  );
}

function isFailureSummary(value: unknown): value is FailureSummary {
  return (
    isRecord(value) &&
    Array.isArray(value.groups) &&
    value.groups.every(isFailureGroup) &&
    (value.expiry === null ||
      (isRecord(value.expiry) &&
        typeof value.expiry.urls_with_expiry === "number" &&
        typeof value.expiry.expired_urls === "number" &&
        typeof value.expiry.all_expired === "boolean" &&
        (value.expiry.latest_expired_at === null ||
          typeof value.expiry.latest_expired_at === "number")))
  );
}

export function isConvertError(value: unknown): value is ConvertError {
  return (
    isRecord(value) &&
    (value.kind === "conversion_failed" || value.kind === "download_failed") &&
    typeof value.message === "string" &&
    (value.failure_summary === null ||
      isFailureSummary(value.failure_summary))
  );
}

export function toConvertError(value: unknown): ConvertError {
  if (isConvertError(value)) {
    return value;
  }

  const message =
    value instanceof Error
      ? value.message
      : typeof value === "string"
        ? value
        : "Conversion failed.";
  return {
    kind: "conversion_failed",
    message,
    failure_summary: null,
  };
}

function plural(count: number, singular: string, pluralForm = `${singular}s`) {
  return count === 1 ? singular : pluralForm;
}

function safeExample(value: string): string {
  const basename = value.replace(/\\/g, "/").split("/").pop() || "unknown file";
  return basename.split(/[?#]/, 1)[0] || "unknown file";
}

function breakdownLabel(group: FailureGroup): string {
  const { count } = group;
  switch (group.kind) {
    case "missing_url":
      return `${count} ${plural(count, "image")} had no download link`;
    case "expired_url":
      return `${count} download ${plural(count, "link")} expired`;
    case "access_denied":
      return `${count} ${plural(count, "download")} ${count === 1 ? "expired or was denied" : "expired or were denied"}`;
    case "not_found":
      return `${count} ${plural(count, "image")} ${count === 1 ? "was" : "were"} not found`;
    case "timeout":
      return `${count} ${plural(count, "download")} timed out`;
    case "connect":
      return `${count} ${plural(count, "connection")} failed`;
    case "dns":
      return `${count} image server ${plural(count, "address", "addresses")} could not be found`;
    case "blocked_address":
      return `${count} ${plural(count, "download")} stopped for safety`;
    case "malformed_url":
      return `${count} download ${plural(count, "link")} ${count === 1 ? "was" : "were"} invalid`;
    case "unsupported_scheme":
      return `${count} download ${plural(count, "link")} used an unsupported format`;
    case "server_error":
      return `${count} image server ${plural(count, "error")}`;
    case "response_error":
      return `${count} incomplete ${plural(count, "download")}`;
    case "too_large":
      return `${count} ${plural(count, "image")} exceeded 50 MB`;
    case "size_overflow":
      return `${count} ${plural(count, "image")} had an invalid size`;
    case "http_error":
      return `${count} image server ${plural(count, "error")}`;
    case "download_error":
      return `${count} ${plural(count, "download")} failed`;
  }
}

function breakdownRemedy(kind: FailureKind): string {
  switch (kind) {
    case "expired_url":
    case "access_denied":
    case "missing_url":
    case "malformed_url":
    case "unsupported_scheme":
    case "not_found":
    case "size_overflow":
      return "Export the dataset again.";
    case "timeout":
    case "connect":
    case "dns":
    case "response_error":
    case "download_error":
      return "Check your connection and try again.";
    case "blocked_address":
      return "Try a different network or ask your network administrator.";
    case "server_error":
    case "http_error":
      return "Wait a moment and try again.";
    case "too_large":
      return "Use smaller images and try again.";
  }
}

function formatBreakdown(group: FailureGroup): string {
  const examples = group.examples.slice(0, 3).map(safeExample);
  const label = breakdownLabel(group);
  const cause =
    examples.length > 0 ? `${label} — ${examples.join(", ")}` : label;
  return `${cause}. ${breakdownRemedy(group.kind)}`;
}

function skippedSentence(count: number): string {
  const image = plural(count, "image");
  const annotation = plural(count, "annotation");
  const pronoun = count === 1 ? "its" : "their";
  return `${count} ${image} and ${pronoun} corresponding ${annotation} were skipped.`;
}

function causeSentence(
  kind: FailureKind,
  count: number,
  hardFailure: boolean,
): string | null {
  switch (kind) {
    case "expired_url":
      return hardFailure
        ? "Your download links have expired. Export the dataset again from Ultralytics Platform, then retry."
        : "Their download links expired. Export the dataset again from Ultralytics Platform, then retry.";
    case "access_denied":
      return `${count} ${plural(count, "image")} could not be downloaded because ${count === 1 ? "its link" : "their links"} expired or access was denied. Export the dataset again.`;
    case "blocked_address":
      return "On this network, the image server points to a local address, so the app stopped the download for safety. Try a different network, or ask your network administrator.";
    case "not_found":
      return `${count} ${plural(count, "image")} could not be found on the server. Export the dataset again. If it keeps happening, ${count === 1 ? "that image may" : "those images may"} have been deleted from the dataset.`;
    case "timeout":
      return `${count} ${plural(count, "download")} timed out. Your connection may be slow. Try again.`;
    case "server_error":
      return "The image server had a problem. Wait a moment and try again.";
    case "too_large":
      return `${count} ${plural(count, "image")} ${count === 1 ? "was" : "were"} larger than 50 MB. Use smaller images and try again.`;
    case "missing_url":
      return `${count} ${plural(count, "image")} had no download link. Export the dataset again.`;
    case "malformed_url":
      return `${count} download ${plural(count, "link")} ${count === 1 ? "was" : "were"} invalid. Export the dataset again and retry.`;
    case "unsupported_scheme":
      return `${count} download ${plural(count, "link")} used an unsupported format. Export the dataset again and retry.`;
    case "response_error":
      return `${count} ${plural(count, "download")} stopped before finishing. Check your connection and try again.`;
    case "size_overflow":
      return `${count} ${plural(count, "image")} reported an invalid size. Export the dataset again and retry.`;
    case "http_error":
      return `The image server rejected ${count} ${plural(count, "download")}. Export the dataset again or try later.`;
    case "download_error":
      return `${count} ${plural(count, "download")} failed. Check your internet connection and try again.`;
    case "connect":
    case "dns":
      return "Couldn't connect to the image server. Check your internet connection and try again.";
  }
}

function dominantGroup(groups: FailureGroup[]): FailureGroup | null {
  let dominant: FailureGroup | null = null;
  let tied = false;
  for (const group of groups) {
    if (!dominant || group.count > dominant.count) {
      dominant = group;
      tied = false;
    } else if (group.count === dominant.count) {
      tied = true;
    }
  }
  return tied ? null : dominant;
}

function formatExpiryDate(timestamp: number): string | null {
  const date = new Date(timestamp * 1000);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

export function buildDownloadFailureMessage(
  summary: FailureSummary,
  skippedImages: number,
  hardFailure: boolean,
): DownloadFailureMessage | null {
  const groups = summary.groups.filter((group) => group.count > 0);
  if (groups.length === 0 && skippedImages === 0) {
    return null;
  }

  const total = groups.reduce((sum, group) => sum + group.count, 0);
  const allReachabilityFailures =
    groups.length > 0 &&
    groups.every((group) => group.kind === "connect" || group.kind === "dns");
  const dominant = dominantGroup(groups);
  const expiryDate =
    summary.expiry?.latest_expired_at == null
      ? null
      : formatExpiryDate(summary.expiry.latest_expired_at);
  const cause = summary.expiry?.all_expired
    ? expiryDate
      ? `Your download links expired on ${expiryDate}. Export the dataset again from Ultralytics Platform, then retry.`
      : causeSentence("expired_url", summary.expiry.expired_urls, hardFailure)
    : allReachabilityFailures
      ? "Couldn't connect to the image server. Check your internet connection and try again."
      : groups.length === 1 && dominant
        ? causeSentence(dominant.kind, dominant.count, hardFailure)
        : null;
  const genericCount = skippedImages || total;
  const generic = hardFailure
    ? `${genericCount} ${plural(genericCount, "image")} could not be downloaded.`
    : skippedSentence(genericCount);
  const primary = cause
    ? hardFailure
      ? cause
      : `${generic} ${cause}`
    : `${generic} See details below for causes and next steps.`;
  const breakdown = groups.map(formatBreakdown);
  if (summary.expiry?.all_expired) {
    breakdown.push(
      "If this export is new, check that your computer's date and time are correct.",
    );
  }
  let diagnostics = groups
    .map((group) => {
      const statuses =
        group.http_statuses.length > 0
          ? `; HTTP ${group.http_statuses.join(", ")}`
          : "";
      const examples = group.examples.slice(0, 3).map(safeExample);
      return `${group.kind}: ${group.count}${statuses}${examples.length > 0 ? `; ${examples.join(", ")}` : ""}`;
    })
    .join("\n");
  if (summary.expiry?.latest_expired_at != null) {
    diagnostics += `${diagnostics ? "\n" : ""}latest_expired_at: ${summary.expiry.latest_expired_at}`;
  }

  return { primary, breakdown, diagnostics };
}
