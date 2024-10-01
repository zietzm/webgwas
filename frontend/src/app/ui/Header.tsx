import Link from "next/link";

const Header = () => {
  return (
    <header className="bg-blue-main text-white">
      <div className="container mx-auto px-4 py-6 flex justify-between items-center">
        <Link href="/" className="text-2xl font-bold">
          WebGWAS
        </Link>
        <nav>
          <ul className="flex space-x-6">
            <li>
              <Link href="/" className="">
                Home
              </Link>
            </li>
            <li>
              <Link href="/tree" className="">
                Tree
              </Link>
            </li>
            <li>
              <Link href="/about" className="">
                About
              </Link>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
