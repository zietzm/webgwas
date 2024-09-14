import React, { useState, useMemo } from "react";
import Select, {
  Props as SelectProps,
  ActionMeta,
  InputActionMeta,
  OnChangeValue,
} from "react-select";
import Fuse, { IFuseOptions } from "fuse.js";

interface FuzzySelectProps<OptionType, IsMulti extends boolean = false>
  extends SelectProps<OptionType, IsMulti> {
  options: OptionType[];
  fuseOptions?: IFuseOptions<OptionType>;
  fuseKeys?: Array<keyof OptionType>;
  fuseThreshold?: number;
}

function FuzzySelect<OptionType, IsMulti extends boolean = false>({
  options,
  fuseOptions = {},
  fuseKeys = ["name", "code"] as Array<keyof OptionType>,
  fuseThreshold = 0.3,
  onChange,
  ...props
}: FuzzySelectProps<OptionType, IsMulti>) {
  const [inputValue, setInputValue] = useState("");

  const fuse = useMemo(() => {
    return new Fuse(options || [], {
      keys: fuseKeys as string[],
      threshold: fuseThreshold,
      ...fuseOptions,
    });
  }, [options, fuseKeys, fuseThreshold, fuseOptions]);

  const filteredOptions = useMemo(() => {
    if (!inputValue) {
      return options;
    }
    const results = fuse.search(inputValue);
    return results.map((result) => result.item);
  }, [inputValue, fuse, options]);

  const handleInputChange = (value: string, actionMeta: InputActionMeta) => {
    if (actionMeta.action === "input-change") {
      setInputValue(value);
    }
  };

  const handleChange = (
    selectedOption: OnChangeValue<OptionType, IsMulti>,
    actionMeta: ActionMeta<OptionType>,
  ) => {
    // Clear the input value upon selection
    setInputValue("");

    // Call the parent onChange handler if provided
    if (onChange) {
      onChange(selectedOption, actionMeta);
    }
  };

  return (
    <Select
      {...props}
      options={filteredOptions}
      inputValue={inputValue}
      onInputChange={handleInputChange}
      onChange={handleChange}
      filterOption={null} // Disable default filtering
    />
  );
}

export default FuzzySelect;
