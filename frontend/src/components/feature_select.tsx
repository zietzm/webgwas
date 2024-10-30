import React from "react";

import { ActionMeta, SingleValue } from "react-select";

import FuzzySelect from "@/app/ui/FuzzySelect";
import { Feature } from "@/app/lib/types";

export default function FeatureSelect({
  features,
  onChange,
}: {
  features: Feature[];
  onChange:
    | ((
        newValue: SingleValue<Feature>,
        actionMeta: ActionMeta<Feature>,
      ) => void)
    | undefined;
}) {
  return (
    <FuzzySelect
      fuseThreshold={0.3} // Higher number is more lenient
      options={features}
      closeMenuOnSelect={false}
      onChange={onChange}
      value={null}
      getOptionLabel={(option) =>
        `${option.name} [${option.code}] (N=${option.sample_size})`
      }
      getOptionValue={(option) => option.code}
      placeholder="Search for a field..."
    />
  );
}
