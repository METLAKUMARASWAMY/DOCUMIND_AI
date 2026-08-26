import { zodResolver } from "@hookform/resolvers/zod";
import { Alert, Box, Button, Link, TextField, Typography } from "@mui/material";
import { LogIn } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Link as RouterLink, useLocation, useNavigate } from "../../router";
import { z } from "zod";

import { useAuth } from "../../auth/AuthContext";
import { AuthLayout } from "./AuthLayout";

const schema = z.object({
  email: z.string().email("Enter a valid email address"),
  password: z.string().min(1, "Password is required"),
});
type FormValues = z.infer<typeof schema>;

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [error, setError] = useState("");
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  const submit = async (values: FormValues) => {
    setError("");
    try {
      await login(values.email, values.password);
      const target = (location.state as { from?: string } | null)?.from || "/dashboard";
      navigate(target, { replace: true });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Sign in failed");
    }
  };

  return (
    <AuthLayout>
      <Box component="form" className="auth-form" onSubmit={handleSubmit(submit)} noValidate>
        <Box>
          <Typography variant="h1">Sign in</Typography>
          <Typography color="text.secondary">
            Access your organisation's document workspace.
          </Typography>
        </Box>
        {error ? <Alert severity="error">{error}</Alert> : null}
        <TextField
          label="Work email"
          type="email"
          autoComplete="email"
          fullWidth
          {...register("email")}
          error={Boolean(errors.email)}
          helperText={errors.email?.message}
        />
        <TextField
          label="Password"
          type="password"
          autoComplete="current-password"
          fullWidth
          {...register("password")}
          error={Boolean(errors.password)}
          helperText={errors.password?.message}
        />
        <Box className="auth-row">
          <Link component={RouterLink} to="/forgot-password">
            Forgot password?
          </Link>
        </Box>
        <Button
          type="submit"
          variant="contained"
          size="large"
          startIcon={<LogIn size={18} />}
          disabled={isSubmitting}
        >
          {isSubmitting ? "Signing in" : "Sign in"}
        </Button>
        <Typography color="text.secondary" textAlign="center">
          New to DocuMind?{" "}
          <Link component={RouterLink} to="/signup">
            Create an account
          </Link>
        </Typography>
      </Box>
    </AuthLayout>
  );
}
