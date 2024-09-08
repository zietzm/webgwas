import Link from "next/link";

const Header = () => {
  return (
    <header className="bg-indigo-600 text-white">
      <div className="container mx-auto px-4 py-6 flex justify-between items-center">
        <Link href="/" className="text-2xl font-bold">
          WebGWAS
        </Link>
        <nav>
          <ul className="flex space-x-6">
            <li>
              <Link href="/" className="hover:text-indigo-200">
                Simple
              </Link>
            </li>
            <li>
              <Link href="/complex" className="hover:text-indigo-200">
                Complex
              </Link>
            </li>
            <li>
              <Link href="/about" className="hover:text-indigo-200">
                About
              </Link>
            </li>
            <li>
              <Link href="/citing" className="hover:text-indigo-200">
                Citing
              </Link>
            </li>
            <li>
              <Link href="/faq" className="hover:text-indigo-200">
                FAQ
              </Link>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
