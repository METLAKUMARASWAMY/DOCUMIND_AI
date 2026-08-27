import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Vitest runs without `globals: true`, so Testing Library cannot register its
// own automatic cleanup. Without this, rendered trees accumulate across cases
// in a file and a query can match an element left behind by an earlier test.
afterEach(cleanup);
