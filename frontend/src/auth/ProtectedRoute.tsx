import { Box, CircularProgress } from "@mui/material";
import { Navigate, Outlet, useLocation } from "../router";

import { useAuth } from "./AuthContext";

export function ProtectedRoute() {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) {
    return (
      <Box className="full-page-center" aria-label="Loading account">
        <CircularProgress size={28} />
      </Box>
    );
  }
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return <Outlet />;
}
