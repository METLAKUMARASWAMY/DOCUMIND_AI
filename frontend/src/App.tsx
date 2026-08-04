import { lazy, Suspense } from "react";

import { ProtectedRoute } from "./auth/ProtectedRoute";
import { AppShell } from "./components/AppShell";
import { LoadingState } from "./components/Feedback";
import { Navigate, Route, Routes } from "./router";

const AdminSettingsPage = lazy(() => import("./pages/AdminSettingsPage").then((module) => ({ default: module.AdminSettingsPage })));
const AnalyticsPage = lazy(() => import("./pages/AnalyticsPage").then((module) => ({ default: module.AnalyticsPage })));
const AskDocumentsPage = lazy(() => import("./pages/AskDocumentsPage").then((module) => ({ default: module.AskDocumentsPage })));
const AuditHistoryPage = lazy(() => import("./pages/AuditHistoryPage").then((module) => ({ default: module.AuditHistoryPage })));
const DashboardPage = lazy(() => import("./pages/DashboardPage").then((module) => ({ default: module.DashboardPage })));
const DocumentLibraryPage = lazy(() => import("./pages/DocumentLibraryPage").then((module) => ({ default: module.DocumentLibraryPage })));
const DocumentViewerPage = lazy(() => import("./pages/DocumentViewerPage").then((module) => ({ default: module.DocumentViewerPage })));
const FinanceNarrativesPage = lazy(() => import("./pages/FinanceNarrativesPage").then((module) => ({ default: module.FinanceNarrativesPage })));
const GraphExplorerPage = lazy(() => import("./pages/GraphExplorerPage").then((module) => ({ default: module.GraphExplorerPage })));
const ModelLabPage = lazy(() => import("./pages/ModelLabPage").then((module) => ({ default: module.ModelLabPage })));
const ProcessingStatusPage = lazy(() => import("./pages/ProcessingStatusPage").then((module) => ({ default: module.ProcessingStatusPage })));
const ReportsPage = lazy(() => import("./pages/ReportsPage").then((module) => ({ default: module.ReportsPage })));
const ReviewQueuePage = lazy(() => import("./pages/ReviewQueuePage").then((module) => ({ default: module.ReviewQueuePage })));
const ReviewWorkspacePage = lazy(() => import("./pages/ReviewWorkspacePage").then((module) => ({ default: module.ReviewWorkspacePage })));
const UploadWizardPage = lazy(() => import("./pages/UploadWizardPage").then((module) => ({ default: module.UploadWizardPage })));
const ForgotPasswordPage = lazy(() => import("./pages/auth/ForgotPasswordPage").then((module) => ({ default: module.ForgotPasswordPage })));
const LoginPage = lazy(() => import("./pages/auth/LoginPage").then((module) => ({ default: module.LoginPage })));
const SignupPage = lazy(() => import("./pages/auth/SignupPage").then((module) => ({ default: module.SignupPage })));

export default function App() {
  return (
    <Suspense fallback={<LoadingState label="Loading workspace" />}>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        <Route element={<ProtectedRoute />}>
          <Route element={<AppShell />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/upload" element={<UploadWizardPage />} />
            <Route path="/processing/:jobId" element={<ProcessingStatusPage />} />
            <Route path="/documents" element={<DocumentLibraryPage />} />
            <Route path="/documents/:documentId" element={<DocumentViewerPage />} />
            <Route path="/reviews" element={<ReviewQueuePage />} />
            <Route path="/reviews/:reviewId" element={<ReviewWorkspacePage />} />
            <Route path="/ask" element={<AskDocumentsPage />} />
            <Route path="/graph" element={<GraphExplorerPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/narratives" element={<FinanceNarrativesPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/model-lab" element={<ModelLabPage />} />
            <Route path="/admin" element={<AdminSettingsPage />} />
            <Route path="/audit" element={<AuditHistoryPage />} />
          </Route>
        </Route>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Suspense>
  );
}
