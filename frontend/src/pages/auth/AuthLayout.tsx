import { Box, Typography } from "@mui/material";
import { BookOpenCheck, CheckCircle2 } from "lucide-react";

export function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <Box className="auth-layout">
      <Box className="auth-visual" role="presentation">
        <img src="/auth-document-workspace.jpg" alt="Document intelligence review workspace" />
        <Box className="auth-visual-overlay">
          <Box className="auth-brand">
            <BookOpenCheck size={26} />
            <Typography variant="h2">DocuMind Vision AI</Typography>
          </Box>
          <Box className="auth-proof-list">
            {["Visual evidence retained", "Human-verified decisions", "Grounded document answers"].map((item) => (
              <Box key={item} className="auth-proof">
                <CheckCircle2 size={17} />
                <Typography>{item}</Typography>
              </Box>
            ))}
          </Box>
        </Box>
      </Box>
      <Box className="auth-form-wrap">{children}</Box>
    </Box>
  );
}
