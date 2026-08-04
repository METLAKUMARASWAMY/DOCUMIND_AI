import { zodResolver } from "@hookform/resolvers/zod";
import { Alert, Box, Button, Link, TextField, Typography } from "@mui/material";
import { UserPlus } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Link as RouterLink, useNavigate } from "../../router";
import { z } from "zod";

import { api } from "../../api/client";
import { AuthLayout } from "./AuthLayout";

const schema = z
  .object({
    name: z.string().min(2, "Name is required").max(120),
    organisation_name: z.string().min(2, "Organisation is required"),
    email: z.string().email("Enter a valid email address"),
    password: z
      .string()
      .min(12, "Use at least 12 characters")
      .regex(/[A-Z]/, "Include an uppercase letter")
      .regex(/[a-z]/, "Include a lowercase letter")
      .regex(/\d/, "Include a number")
      .regex(/[^A-Za-z0-9]/, "Include a symbol"),
    confirmPassword: z.string(),
  })
  .refine((values) => values.password === values.confirmPassword, {
    path: ["confirmPassword"],
    message: "Passwords do not match",
  });
type FormValues = z.infer<typeof schema>;

export function SignupPage() {
  const navigate = useNavigate();
  const [error, setError] = useState("");
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  const submit = async ({ confirmPassword: _, ...values }: FormValues) => {
    setError("");
    try {
      await api.register(values);
      navigate("/login", { replace: true, state: { registered: true } });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Registration failed");
    }
  };

  return (
    <AuthLayout>
      <Box component="form" className="auth-form auth-form-wide" onSubmit={handleSubmit(submit)} noValidate>
        <Box>
          <Typography variant="h1">Create account</Typography>
          <Typography color="text.secondary">New accounts begin with Viewer access.</Typography>
        </Box>
        {error ? <Alert severity="error">{error}</Alert> : null}
        <Box className="two-column-fields">
          <TextField label="Full name" {...register("name")} error={Boolean(errors.name)} helperText={errors.name?.message} />
          <TextField
            label="Organisation"
            {...register("organisation_name")}
            error={Boolean(errors.organisation_name)}
            helperText={errors.organisation_name?.message}
          />
        </Box>
        <TextField label="Work email" type="email" {...register("email")} error={Boolean(errors.email)} helperText={errors.email?.message} />
        <Box className="two-column-fields">
          <TextField label="Password" type="password" {...register("password")} error={Boolean(errors.password)} helperText={errors.password?.message} />
          <TextField
            label="Confirm password"
            type="password"
            {...register("confirmPassword")}
            error={Boolean(errors.confirmPassword)}
            helperText={errors.confirmPassword?.message}
          />
        </Box>
        <Button type="submit" variant="contained" size="large" startIcon={<UserPlus size={18} />} disabled={isSubmitting}>
          {isSubmitting ? "Creating account" : "Create account"}
        </Button>
        <Typography color="text.secondary" textAlign="center">
          Already registered? <Link component={RouterLink} to="/login">Sign in</Link>
        </Typography>
      </Box>
    </AuthLayout>
  );
}
