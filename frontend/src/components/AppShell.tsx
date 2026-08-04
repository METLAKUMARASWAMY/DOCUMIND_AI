import {
  AppBar,
  Avatar,
  Box,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Toolbar,
  Tooltip,
  Typography,
  useMediaQuery,
} from "@mui/material";
import {
  BarChart3,
  BookOpenCheck,
  ClipboardCheck,
  FileBarChart,
  FileClock,
  Files,
  FlaskConical,
  LayoutDashboard,
  LogOut,
  Menu as MenuIcon,
  MessageSquareText,
  Network,
  Settings,
  ShieldCheck,
  Upload,
  WalletCards,
  type LucideIcon,
} from "lucide-react";
import { useMemo, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "../router";

import { useAuth } from "../auth/AuthContext";
import type { UserRole } from "../types/api";

const drawerWidth = 238;

interface NavigationItem {
  path: string;
  label: string;
  icon: LucideIcon;
  roles?: UserRole[];
}

const navigation: NavigationItem[] = [
  { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { path: "/upload", label: "Upload", icon: Upload, roles: ["ADMIN", "PROCESSOR", "REVIEWER", "SENIOR_REVIEWER", "COMPLIANCE_OFFICER"] },
  { path: "/documents", label: "Documents", icon: Files },
  { path: "/reviews", label: "Review queue", icon: ClipboardCheck, roles: ["ADMIN", "REVIEWER", "SENIOR_REVIEWER", "COMPLIANCE_OFFICER"] },
  { path: "/ask", label: "Ask documents", icon: MessageSquareText },
  { path: "/graph", label: "Knowledge graph", icon: Network },
  { path: "/analytics", label: "Analytics", icon: BarChart3 },
  { path: "/narratives", label: "Finance narratives", icon: WalletCards },
  { path: "/reports", label: "Reports", icon: FileBarChart },
  { path: "/model-lab", label: "Model laboratory", icon: FlaskConical },
  { path: "/admin", label: "Admin settings", icon: Settings, roles: ["ADMIN"] },
  { path: "/audit", label: "Audit history", icon: FileClock, roles: ["ADMIN", "AUDITOR"] },
];

export function AppShell() {
  const { user, logout } = useAuth();
  const mobile = useMediaQuery("(max-width:900px)");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const visibleNavigation = useMemo(
    () => navigation.filter((item) => !item.roles || (user && item.roles.includes(user.role))),
    [user],
  );

  const drawer = (
    <Box className="app-drawer">
      <Box className="brand-lockup">
        <Box className="brand-mark" aria-hidden="true">
          <BookOpenCheck size={21} />
        </Box>
        <Box>
          <Typography className="brand-name">DocuMind</Typography>
          <Typography className="brand-subtitle">Vision AI</Typography>
        </Box>
      </Box>
      <Divider />
      <List className="nav-list">
        {visibleNavigation.map(({ path, label, icon: Icon }) => (
          <ListItemButton
            component={NavLink}
            to={path}
            key={path}
            selected={location.pathname === path || location.pathname.startsWith(`${path}/`)}
            onClick={() => setDrawerOpen(false)}
          >
            <ListItemIcon>
              <Icon size={18} />
            </ListItemIcon>
            <ListItemText primary={label} />
          </ListItemButton>
        ))}
      </List>
      <Box className="tenant-block">
        <ShieldCheck size={16} />
        <Box>
          <Typography variant="caption" color="text.secondary">
            Organisation
          </Typography>
          <Typography variant="body2" noWrap>
            {user?.tenant_id.replace("tenant-", "")}
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  const handleLogout = async () => {
    setAnchor(null);
    await logout();
    navigate("/login", { replace: true });
  };

  return (
    <Box className="app-frame">
      {mobile ? (
        <Drawer open={drawerOpen} onClose={() => setDrawerOpen(false)} PaperProps={{ sx: { width: drawerWidth } }}>
          {drawer}
        </Drawer>
      ) : (
        <Drawer variant="permanent" PaperProps={{ sx: { width: drawerWidth } }}>
          {drawer}
        </Drawer>
      )}
      <AppBar position="fixed" color="inherit" elevation={0} className="topbar" sx={{ left: mobile ? 0 : drawerWidth }}>
        <Toolbar>
          {mobile ? (
            <Tooltip title="Open navigation">
              <IconButton onClick={() => setDrawerOpen(true)} aria-label="Open navigation">
                <MenuIcon size={20} />
              </IconButton>
            </Tooltip>
          ) : null}
          <Box sx={{ flex: 1 }} />
          <Tooltip title="Account menu">
            <IconButton onClick={(event) => setAnchor(event.currentTarget)} aria-label="Account menu">
              <Avatar sx={{ width: 32, height: 32, bgcolor: "secondary.main", fontSize: 13 }}>
                {user?.name
                  .split(" ")
                  .slice(0, 2)
                  .map((part) => part[0])
                  .join("")}
              </Avatar>
            </IconButton>
          </Tooltip>
          <Menu anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)}>
            <Box className="account-menu-header">
              <Typography variant="body2" fontWeight={700}>
                {user?.name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {user?.role}
              </Typography>
            </Box>
            <Divider />
            <MenuItem onClick={() => void handleLogout()}>
              <ListItemIcon>
                <LogOut size={17} />
              </ListItemIcon>
              Sign out
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>
      <Box component="main" className="app-main" sx={{ marginLeft: mobile ? 0 : `${drawerWidth}px` }}>
        <Outlet />
      </Box>
    </Box>
  );
}
