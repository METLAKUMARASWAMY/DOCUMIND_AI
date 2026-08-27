/**
 * Client-side pre-flight for the upload queue.
 *
 * The backend remains the authority — it enforces MAX_UPLOAD_BYTES and rejects
 * unreadable content regardless of what happens here. This exists so a user
 * learns that a 60 MB TIFF will be refused before waiting for the upload, not
 * after. Every limit below is stated in the UI so the two never drift silently.
 */

/** Mirrors MAX_UPLOAD_BYTES in backend/app/core/config.py. */
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

/** Mirrors the accept list the ingestion service will parse. */
export const ACCEPTED_EXTENSIONS = ["pdf", "txt", "png", "jpg", "jpeg", "tif", "tiff"] as const;

export const MAX_FILES_PER_JOB = 20;

export type RejectionReason = "unsupported_type" | "too_large" | "empty" | "duplicate" | "job_full";

export interface Rejection {
  file: File;
  reason: RejectionReason;
  message: string;
}

export interface ValidationResult {
  accepted: File[];
  rejected: Rejection[];
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot + 1).toLowerCase();
}

/** Identity used for duplicate detection within a single queue. */
export function fileKey(file: File): string {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

/**
 * Filters a set of newly-selected files against the files already queued.
 *
 * Returns both halves: callers need the rejections to explain themselves, and
 * silently dropping a file the user selected is its own kind of dishonesty.
 */
export function validateSelection(selected: File[], existing: File[]): ValidationResult {
  const seen = new Set(existing.map(fileKey));
  const accepted: File[] = [];
  const rejected: Rejection[] = [];
  let slots = MAX_FILES_PER_JOB - existing.length;

  for (const file of selected) {
    const extension = extensionOf(file.name);

    if (!(ACCEPTED_EXTENSIONS as readonly string[]).includes(extension)) {
      rejected.push({
        file,
        reason: "unsupported_type",
        message: `${file.name} is a .${extension || "unknown"} file. Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}.`,
      });
      continue;
    }
    if (file.size === 0) {
      rejected.push({ file, reason: "empty", message: `${file.name} is empty.` });
      continue;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      rejected.push({
        file,
        reason: "too_large",
        message: `${file.name} is ${formatBytes(file.size)}. The limit is ${formatBytes(MAX_UPLOAD_BYTES)}.`,
      });
      continue;
    }
    const key = fileKey(file);
    if (seen.has(key)) {
      rejected.push({
        file,
        reason: "duplicate",
        message: `${file.name} is already in this job.`,
      });
      continue;
    }
    if (slots <= 0) {
      rejected.push({
        file,
        reason: "job_full",
        message: `${file.name} was not added: a job holds at most ${MAX_FILES_PER_JOB} files.`,
      });
      continue;
    }

    seen.add(key);
    slots -= 1;
    accepted.push(file);
  }

  return { accepted, rejected };
}
