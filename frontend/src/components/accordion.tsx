import React, { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

export default function Accordion({
  title,
  children,
}: Readonly<{ title: string; children: React.ReactNode }>) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div>
      <h2>
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className={`flex items-center justify-between w-full p-5 bg-white border dark:border-gray-700 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 gap-3 ${isOpen ? "rounded-t-lg" : "rounded-lg"}`}
        >
          <span>{title}</span>
          {isOpen ? (
            <ChevronUp className="font-bold w-5 h-5 shrink-0" />
          ) : (
            <ChevronDown className="font-bold w-5 h-5 shrink-0" />
          )}
        </button>
      </h2>
      {isOpen && (
        <div className="p-5 border border-t-0 rounded-b-lg bg-white dark:bg-gray-800">
          {children}
        </div>
      )}
    </div>
  );
}
