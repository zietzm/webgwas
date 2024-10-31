"use client";

import { Button } from "@/components/ui/button";
import { Menu, X } from "lucide-react";
import Link from "next/link";
import Image from "next/image";
import React, { useState } from "react";
import { usePathname } from "next/navigation";

const sections = [
  {
    name: "Simple",
    href: "/simple",
  },
  {
    name: "Advanced",
    href: "/tree",
  },
  {
    name: "About",
    href: "/about",
  },
];

export default function Header() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="bg-blue-main text-white px-4 relative">
      <div className="container mx-auto max-w-4xl top-0 flex-1">
        <div className="mx-auto py-4 md:py-0 md:px-4 flex justify-between items-center">
          <Link
            href="/"
            className="flex items-center gap-2 text-2xl text-white font-bold hover:text-white no-underline"
          >
            <Image
              src="/logo.webp"
              alt="WebGWAS logo"
              width={50}
              height={50}
              unoptimized
            />
            WebGWAS
          </Link>
          {/* Desktop navigation */}
          <div className="hidden md:flex">
            <nav className="py-4 flex gap-4">
              {sections.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`px-2 py-2 rounded-md text-lg no-underline ${
                    pathname === item.href
                      ? "bg-white text-blue-main hover:text-gray-900"
                      : "text-white hover:bg-blue-dark hover:text-white"
                  }`}
                >
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>

          <div className="md:hidden flex items-center justify-between relative">
            {/* Menu/X Button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setOpen(!open)}
              className="z-20 mx-2"
            >
              {open ? (
                <X className="h-8 w-8 text-white" />
              ) : (
                <Menu className="h-8 w-8 text-white" />
              )}
            </Button>
          </div>
        </div>
        {/* Navigation Sheet */}
        {open && (
          <div className="md:hidden absolute top-full left-0 w-full bg-white z-10 shadow-md">
            <nav className="flex flex-col space-y-4 p-4">
              {sections.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-md font-medium ${
                    pathname === item.href ? "bg-gray-100" : ""
                  }`}
                  onClick={() => setOpen(false)}
                >
                  {item.name}
                </Link>
              ))}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}
