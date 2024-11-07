import React, { useState, useEffect } from "react";

import { Loader } from "lucide-react";

import { Cohort } from "@/app/lib/types";
import { fetchCohorts } from "@/app/lib/api";

export default function CohortSelector({
  selectedCohort,
  setSelectedCohort,
  isMutable,
}: {
  selectedCohort: Cohort | null;
  setSelectedCohort: React.Dispatch<React.SetStateAction<Cohort | null>>;
  isMutable: boolean;
}) {
  const [cohorts, setCohorts] = useState<Cohort[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const doFetch = async () => {
      setIsLoading(true);
      try {
        const cohorts = await fetchCohorts();
        setCohorts(cohorts);
        setSelectedCohort(cohorts[0]);
      } catch (error) {
        let errorMessage = "Error fetching cohorts";
        if (error instanceof Error) {
          errorMessage = `Error fetching cohorts: ${error.message}`;
        }
        setError(errorMessage);
      }
      setIsLoading(false);
    };
    doFetch();
  }, [setSelectedCohort]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader className="animate-spin" size={40} />
      </div>
    );
  }
  if (error) {
    return <div className="text-red-600 text-center">{error}</div>;
  }

  return (
    <div className="flex flex-wrap gap-2">
      {cohorts.map((cohort) => (
        <button
          key={cohort.id}
          onClick={() => (isMutable ? setSelectedCohort(cohort) : null)}
          className={`py-2 px-4 rounded-full ${
            selectedCohort === cohort
              ? "bg-primary text-primary-foreground"
              : "bg-gray-200 text-gray-800"
          } ${isMutable ? "cursor-pointer transition-colors " : "cursor-not-allowed"} ${isMutable && selectedCohort === cohort ? "hover:bg-blue-dark" : ""} ${
            isMutable && selectedCohort !== cohort ? "hover:bg-gray-300" : ""
          }`}
        >
          {cohort.name}
        </button>
      ))}
    </div>
  );
}
