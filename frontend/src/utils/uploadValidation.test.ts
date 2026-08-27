import { describe, expect, it } from "vitest";

import {
  MAX_FILES_PER_JOB,
  MAX_UPLOAD_BYTES,
  fileKey,
  formatBytes,
  validateSelection,
} from "./uploadValidation";

function makeFile(name: string, size: number, lastModified = 1_700_000_000_000): File {
  const file = new File(["x"], name, { lastModified });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

describe("validateSelection", () => {
  it("accepts every supported extension", () => {
    const files = ["a.pdf", "b.PNG", "c.jpeg", "d.tiff", "e.txt"].map((n) => makeFile(n, 1024));
    const { accepted, rejected } = validateSelection(files, []);
    expect(accepted).toHaveLength(5);
    expect(rejected).toHaveLength(0);
  });

  it("rejects an unsupported type and says what is accepted", () => {
    const { accepted, rejected } = validateSelection([makeFile("payload.exe", 1024)], []);
    expect(accepted).toHaveLength(0);
    expect(rejected[0].reason).toBe("unsupported_type");
    expect(rejected[0].message).toContain("pdf");
  });

  it("rejects a file with no extension at all", () => {
    const { rejected } = validateSelection([makeFile("README", 512)], []);
    expect(rejected[0].reason).toBe("unsupported_type");
  });

  it("rejects a file over the backend limit and names the limit", () => {
    const { rejected } = validateSelection([makeFile("scan.pdf", MAX_UPLOAD_BYTES + 1)], []);
    expect(rejected[0].reason).toBe("too_large");
    expect(rejected[0].message).toContain("25.00 MB");
  });

  it("accepts a file exactly at the limit", () => {
    const { accepted } = validateSelection([makeFile("scan.pdf", MAX_UPLOAD_BYTES)], []);
    expect(accepted).toHaveLength(1);
  });

  it("rejects an empty file", () => {
    const { rejected } = validateSelection([makeFile("blank.pdf", 0)], []);
    expect(rejected[0].reason).toBe("empty");
  });

  it("detects a duplicate against the files already queued", () => {
    const existing = makeFile("invoice.pdf", 2048);
    const { accepted, rejected } = validateSelection([makeFile("invoice.pdf", 2048)], [existing]);
    expect(accepted).toHaveLength(0);
    expect(rejected[0].reason).toBe("duplicate");
  });

  it("detects a duplicate inside a single selection", () => {
    const { accepted, rejected } = validateSelection(
      [makeFile("invoice.pdf", 2048), makeFile("invoice.pdf", 2048)],
      [],
    );
    expect(accepted).toHaveLength(1);
    expect(rejected[0].reason).toBe("duplicate");
  });

  it("treats a same-named file with different content as distinct", () => {
    const { accepted } = validateSelection(
      [makeFile("invoice.pdf", 4096, 1_700_000_000_001)],
      [makeFile("invoice.pdf", 2048)],
    );
    expect(accepted).toHaveLength(1);
  });

  it("stops at the per-job file cap and explains why", () => {
    const existing = Array.from({ length: MAX_FILES_PER_JOB }, (_, i) =>
      makeFile(`doc-${i}.pdf`, 1024),
    );
    const { accepted, rejected } = validateSelection([makeFile("extra.pdf", 1024)], existing);
    expect(accepted).toHaveLength(0);
    expect(rejected[0].reason).toBe("job_full");
  });

  it("returns rejections rather than silently dropping files", () => {
    const { accepted, rejected } = validateSelection(
      [makeFile("good.pdf", 1024), makeFile("bad.exe", 1024)],
      [],
    );
    expect(accepted.map((f) => f.name)).toEqual(["good.pdf"]);
    expect(rejected.map((r) => r.file.name)).toEqual(["bad.exe"]);
  });
});

describe("fileKey", () => {
  it("distinguishes files that differ only in size", () => {
    expect(fileKey(makeFile("a.pdf", 1))).not.toBe(fileKey(makeFile("a.pdf", 2)));
  });
});

describe("formatBytes", () => {
  it("scales the unit to the magnitude", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.00 MB");
  });
});
