import React from "react";
import { Github, TLab } from "./Icons";

const Footer = () => {
  return (
    <footer className="bg-gray-100 text-gray-600 py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm">
              &copy; 2024 Tatonetti Lab. All rights reserved.
            </p>
          </div>
          <div className="flex space-x-6">
            <a
              href="https://tatonettilab.org/"
              target="_blank"
              className="text-gray-400 hover:text-gray-500"
            >
              <span className="sr-only">TLab</span>
              <TLab className="h-6 w-6" />
            </a>
            <a
              href="https://github.com/tatonetti-lab"
              target="_blank"
              className="text-gray-400 hover:text-gray-500"
            >
              <span className="sr-only">GitHub</span>
              <Github className="h-6 w-6" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
