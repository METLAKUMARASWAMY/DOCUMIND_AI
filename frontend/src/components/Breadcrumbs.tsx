import { Breadcrumbs as MuiBreadcrumbs, Link as MuiLink, Typography } from "@mui/material";
import { ChevronRight } from "lucide-react";
import { useMemo } from "react";

import { Link, useLocation } from "../router";

/**
 * Route segment -> human label. Segments not listed here are rendered as-is
 * (title-cased), so a new route gets a sensible crumb without an edit.
 */
const SEGMENT_LABELS: Record<string, string> = {
  dashboard: "Dashboard",
  upload: "Upload",
  documents: "Documents",
  processing: "Processing",
  reviews: "Review queue",
  ask: "Ask documents",
  graph: "Knowledge graph",
  analytics: "Analytics",
  narratives: "Finance narratives",
  reports: "Reports",
  "model-lab": "Model laboratory",
  admin: "Admin settings",
  audit: "Audit history",
};

/** Segments that are containers, not destinations — rendered without a link. */
const NON_NAVIGABLE = new Set(["processing"]);

function labelFor(segment: string): string {
  const known = SEGMENT_LABELS[segment];
  if (known) return known;
  // An id segment: shorten it so the crumb stays readable.
  if (segment.length > 24) return `${segment.slice(0, 12)}…${segment.slice(-4)}`;
  return segment.charAt(0).toUpperCase() + segment.slice(1).replaceAll("-", " ");
}

export function Breadcrumbs() {
  const { pathname } = useLocation();

  const crumbs = useMemo(() => {
    const segments = pathname.split("/").filter(Boolean);
    return segments.map((segment, index) => ({
      segment,
      label: labelFor(segment),
      href: `/${segments.slice(0, index + 1).join("/")}`,
      last: index === segments.length - 1,
      navigable: !NON_NAVIGABLE.has(segment),
    }));
  }, [pathname]);

  if (crumbs.length <= 1) return null;

  return (
    <MuiBreadcrumbs
      aria-label="Breadcrumb"
      separator={<ChevronRight size={14} aria-hidden="true" />}
      className="app-breadcrumbs"
    >
      <MuiLink component={Link} to="/dashboard" underline="hover" color="text.secondary">
        Home
      </MuiLink>
      {crumbs.map(({ label, href, last, navigable }) =>
        last || !navigable ? (
          <Typography
            key={href}
            color={last ? "text.primary" : "text.secondary"}
            aria-current={last ? "page" : undefined}
          >
            {label}
          </Typography>
        ) : (
          <MuiLink key={href} component={Link} to={href} underline="hover" color="text.secondary">
            {label}
          </MuiLink>
        ),
      )}
    </MuiBreadcrumbs>
  );
}
