import PhenotypeBuilder from "@/app/ui/PhenotypeBuilder";

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-8 text-indigo-800">WebGWAS Phenotype Builder</h1>
      <PhenotypeBuilder />
    </div>
  );
}
