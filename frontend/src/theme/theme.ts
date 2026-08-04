import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#12324a", dark: "#0b2234", light: "#365b72" },
    secondary: { main: "#087f78", dark: "#05625e", light: "#42a49e" },
    error: { main: "#b42318" },
    warning: { main: "#b54708" },
    success: { main: "#18794e" },
    info: { main: "#1769aa" },
    background: { default: "#f4f6f8", paper: "#ffffff" },
    text: { primary: "#18232d", secondary: "#596673" },
    divider: "#dfe4e8",
  },
  shape: { borderRadius: 6 },
  typography: {
    fontFamily: 'Inter, "IBM Plex Sans", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    h1: { fontSize: "2rem", fontWeight: 700, lineHeight: 1.2 },
    h2: { fontSize: "1.5rem", fontWeight: 700, lineHeight: 1.25 },
    h3: { fontSize: "1.125rem", fontWeight: 700, lineHeight: 1.3 },
    h4: { fontSize: "1rem", fontWeight: 700, lineHeight: 1.35 },
    button: { textTransform: "none", fontWeight: 650 },
    body1: { fontSize: "0.925rem" },
    body2: { fontSize: "0.825rem" },
  },
  components: {
    MuiButton: { defaultProps: { disableElevation: true }, styleOverrides: { root: { minHeight: 36 } } },
    MuiCard: { styleOverrides: { root: { border: "1px solid #dfe4e8", boxShadow: "none" } } },
    MuiPaper: { styleOverrides: { rounded: { borderRadius: 6 } } },
    MuiTableCell: { styleOverrides: { head: { fontWeight: 700, color: "#344451", background: "#f7f9fa" } } },
    MuiChip: { styleOverrides: { root: { borderRadius: 5, fontWeight: 650 } } },
    MuiTooltip: { defaultProps: { arrow: true } },
  },
});
