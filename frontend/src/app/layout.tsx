import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Header from "./ui/Header";
import Footer from "./ui/Footer";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "WebGWAS",
  description: "Web-based approximate GWAS",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex flex-col min-h-screen min-w-fit">
          <Header />
          <div className="flex flex-row flex-1">
            <main className="mx-auto p-4 flex-1 max-w-4xl">{children}</main>
          </div>
          <Footer />
        </div>
      </body>
    </html>
  );
}
