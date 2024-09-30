export default function About() {
  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 text-blue-dark">About WebGWAS</h1>
      <div className="prose prose-jet">
        <p>
          WebGWAS is a web-based platform for conducting approximate Genome-Wide
          Association Studies (GWAS). Our tool allows researchers to quickly and
          efficiently analyze genetic data to identify potential associations
          between genetic variants and specific traits or diseases.
        </p>
        <br />
        <h2 className="text-xl font-semibold mb-4">Key Features</h2>
        <ul className="list-disc list-inside">
          <li>User-friendly interface for building complex phenotypes</li>
          <li>Fast and efficient GWAS calculations</li>
          <li>Support for multiple cohorts</li>
        </ul>
        <br />
        <p>
          WebGWAS aims to democratize genetic research by providing a powerful,
          accessible tool for researchers worldwide.
        </p>
      </div>
    </div>
  );
}
