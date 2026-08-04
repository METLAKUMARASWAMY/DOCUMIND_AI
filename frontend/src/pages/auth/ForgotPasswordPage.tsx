import { zodResolver } from "@hookform/resolvers/zod";
import { Alert, Box, Button, Link, TextField, Typography } from "@mui/material";
import { Send } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Link as RouterLink } from "../../router";
import { z } from "zod";

import { api } from "../../api/client";
import { AuthLayout } from "./AuthLayout";

const schema = z.object({ email: z.string().email("Enter a valid email address") });

export function ForgotPasswordPage() {
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
  });

  return (
    <AuthLayout>
      <Box
        component="form"
        className="auth-form"
        onSubmit={handleSubmit(async ({ email }) => {
          setError("");
          try {
            setMessage((await api.forgotPassword(email)).message);
          } catch (reason) {
            setError(reason instanceof Error ? reason.message : "Request failed");
          }
        })}
        noValidate
      >
        <Box>
          <Typography variant="h1">Reset password</Typography>
          <Typography color="text.secondary">Enter the email associated with your account.</Typography>
        </Box>
        {message ? <Alert severity="success">{message}</Alert> : null}
        {error ? <Alert severity="error">{error}</Alert> : null}
        <TextField label="Work email" type="email" {...register("email")} error={Boolean(errors.email)} helperText={errors.email?.message} />
        <Button type="submit" variant="contained" size="large" startIcon={<Send size={18} />} disabled={isSubmitting}>
          Send reset request
        </Button>
        <Link component={RouterLink} to="/login" textAlign="center">Return to sign in</Link>
      </Box>
    </AuthLayout>
  );
}
