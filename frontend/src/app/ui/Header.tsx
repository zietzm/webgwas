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
    href: "/",
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
    <header className="bg-blue-main text-white relative">
      <div className="container mx-auto max-w-4xl top-0 px-4">
        <div className="container mx-auto flex-1">
          <div className="mx-auto py-6 flex justify-between items-center">
            <Link
              href="/"
              className="flex items-center gap-2 text-2xl font-bold"
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
              <nav className="space-x-4 py-4">
                {sections.map((item) => (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`px-3 py-2 rounded-md text-lg ${
                      pathname === item.href
                        ? "bg-white text-blue-main hover:text-gray-900"
                        : "text-white hover:bg-blue-dark"
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
                  <X className="h-8 w-8 text-sky-800" />
                ) : (
                  <Menu className="h-8 w-8 text-sky-800" />
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
      </div>
    </header>
  );
}
