import React from "react";

export default function MainArea({ children }: { children: React.ReactNode }) {
  return (
    <div className="border rounded-lg p-4 md:p-6 flex flex-col gap-4 bg-white dark:bg-gray-800">
      {children}
    </div>
  );
}
