import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "./auth/AuthContext";
import { queryClient } from "./api/query-client";
import { App } from "./app/App";
import { ThemeProvider, ToastProvider } from "./ui";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("App root element is missing.");

createRoot(root).render(
  <StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <ToastProvider>
            <App />
          </ToastProvider>
        </AuthProvider>
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>,
);
