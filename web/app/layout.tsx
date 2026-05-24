import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "GridSentinel — GPU Data-Center Ops + Predictive Maintenance",
  description:
    "Closed-loop GPU fleet operations intelligence: simulator, root-cause clustering, capacity forecast, calibrated PdM model with Cox PH survival, IsolationForest, TCN baseline, MLflow + model registry + drift detection.",
  openGraph: {
    title: "GridSentinel",
    description:
      "GPU data-center operations + PdM platform: 12× alert compression, 0.567 Cox C-index, 2.4× lift@10 on at-risk node ranking.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
