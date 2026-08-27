import { Alert, Snackbar } from "@mui/material";
import { createContext, useCallback, useContext, useMemo, useState } from "react";

export type NotificationTone = "success" | "info" | "warning" | "error";

export interface Notification {
  id: number;
  tone: NotificationTone;
  message: string;
}

interface NotificationContextValue {
  notify: (tone: NotificationTone, message: string) => void;
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

/**
 * Transient confirmation for actions whose result is otherwise invisible.
 *
 * Approving a review, archiving a report or retrying a job all mutate server
 * state and then re-render a list that looks much the same. Without an
 * acknowledgement the user cannot tell a completed action from a swallowed one.
 * Errors are held until dismissed; successes clear themselves.
 */
export function NotificationProvider({ children }: { children: React.ReactNode }) {
  const [queue, setQueue] = useState<Notification[]>([]);

  const notify = useCallback((tone: NotificationTone, message: string) => {
    setQueue((current) => [...current, { id: Date.now() + Math.random(), tone, message }]);
  }, []);

  const dismiss = useCallback((id: number) => {
    setQueue((current) => current.filter((item) => item.id !== id));
  }, []);

  const value = useMemo(() => ({ notify }), [notify]);
  const current = queue[0];

  return (
    <NotificationContext.Provider value={value}>
      {children}
      <Snackbar
        key={current?.id}
        open={Boolean(current)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        autoHideDuration={current?.tone === "error" ? null : 5000}
        onClose={(_event, reason) => {
          if (reason === "clickaway") return;
          if (current) dismiss(current.id);
        }}
      >
        {current ? (
          <Alert
            severity={current.tone}
            variant="filled"
            onClose={() => dismiss(current.id)}
            sx={{ maxWidth: 460 }}
          >
            {current.message}
          </Alert>
        ) : undefined}
      </Snackbar>
    </NotificationContext.Provider>
  );
}

/**
 * Returns a notify function. Safe to call outside the provider — it becomes a
 * no-op rather than throwing, so a component can be unit-tested in isolation.
 */
export function useNotify(): (tone: NotificationTone, message: string) => void {
  const context = useContext(NotificationContext);
  return context?.notify ?? noop;
}

function noop() {
  /* no provider mounted: notifications are a progressive enhancement */
}
