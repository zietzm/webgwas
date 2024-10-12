"use client";

import { Button } from "@/components/ui/button";
import { Menu, X } from "lucide-react";
import Link from "next/link";
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

const Header = () => {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="bg-blue-main text-white relative">
      <div className="container max-w-7xl mx-auto top-0">
        <div className="mx-auto px-4 py-6 flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold">
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
    </header>
  );
};

export default Header;
