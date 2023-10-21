def string_to_dict(input_string):
    # Splitting the string by ', ' to get each key-value pair
    for item in input_string.split(", "):
        key, value = item.split(":")
        dict_output[int(key)] = int(value)


def parse_json_file(filename):
    # Read the file as a string
    with open(filename, "r") as file:
        content = file.read()

    # Remove unnecessary characters
    content = content.replace(" ", "").replace("\n", "").replace('"', "")

    # Split by the outermost curly braces to get the main entries
    main_entries = content[1:-1].split("},")

    parsed_dict = {}

    for entry in main_entries:
        # Handle the last entry which doesn't end with '}'
        if entry[-1] != "}":
            entry += "}"

        # Split each main entry at the first ':' to get the key and its associated dictionary
        key, value_dict_str = entry.split(":", 1)
        value_dict_str = value_dict_str[1:-1]  # remove the curly braces

        # Extract cleaning and type values from the value dictionary
        cleaning_str = value_dict_str.split("Cleaning:[")[1].split("],Type:")[0]
        type_value = value_dict_str.split("Type:")[1]

        # Process the cleaning string
        cleaning_values = cleaning_str.split(",")
        cleaning_list = None
        cleaning_dict = None

        if len(cleaning_values) == 1 and cleaning_values[0] == "":
            cleaning_list = None
            cleaning_dict = None
        else:
            for val in cleaning_values:
                if ":" in val:
                    if cleaning_dict is None:
                        cleaning_dict = {}
                    k, v = map(int, val.split(":"))
                    cleaning_dict[k] = v
                else:
                    if cleaning_list is None:
                        cleaning_list = []
                    cleaning_list.extend(map(int, val.split(",")))

        parsed_dict[key] = ([cleaning_list, cleaning_dict], type_value)

    return parsed_dict
