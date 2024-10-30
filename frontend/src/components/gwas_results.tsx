import React from "react";

import { PvaluesResult } from "@/app/lib/api";
import ManhattanPlot from "@/app/ui/ManhattanPlot";

export default function GwasResults({ results }: { results: PvaluesResult }) {
  return <ManhattanPlot data={results} />;
}
