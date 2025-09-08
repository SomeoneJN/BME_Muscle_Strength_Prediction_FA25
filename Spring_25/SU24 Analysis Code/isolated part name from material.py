import xml.etree.ElementTree as ET


def get_parts_with_material(xml_filepath, material_name):
    """
    Finds all SolidDomain names (part names) that have the given material.

    Args:
        xml_filepath: Path to the XML file.
        material_name: Material name to search for.

    Returns:
        A list of SolidDomain names (part names) or an empty list.
    """
    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        mesh_domains = root.find('MeshDomains')

        part_names = []  # List to store the part names

        if mesh_domains is not None:
            for solid_domain in mesh_domains.findall('SolidDomain'):
                if solid_domain.get('mat') == material_name:
                    part_names.append(solid_domain.get('name'))

        return part_names

    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_filepath}")
        return []
    except ET.ParseError:
        print(f"Error: Could not parse XML file at {xml_filepath}")
        return []

# Example usage:
xml_file = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing\Part1_E(0.01)Part3_E(0.2)Part11_E(0.1).feb"
material = "Levator Ani Material"  # Material to search for

part_names = get_parts_with_material(xml_file, material)
print(part_names)

if part_names:
    print(f"Part names with material '{material}': {part_names}")
else:
    print(f"No parts found with material '{material}'.")
    
    
file_name = r"C:\Users\mgordon\Downloads\Runs for Testing-20250108T014256Z-001\Runs for Testing\Part1_E(0.01)Part3_E(0.2)Part11_E(0.1).feb"

def get_unique_node_numbers(xml_filepath, part_name):
    """
    Finds unique node numbers from the 'elem' elements of a given part.

    Args:
        xml_filepath: Path to the XML file.
        part_name: The name of the part (e.g., "Part10").

    Returns:
        A set of unique node numbers, or an empty set if not found.
    """
    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        elements = root.find('Mesh/Elements[@name="' + part_name + '"]') # More specific find

        if elements is not None:
            node_numbers = set()  # Use a set to store unique numbers
            for elem in elements.findall('elem'):
                nodes_str = elem.text  # Get the text content (node numbers)
                for node in nodes_str.split(','):  # Split by comma and add to set
                    node_numbers.add(int(node)) # Convert to int and add

            return node_numbers
        else:
            print(f"Warning: Elements with name '{part_name}' not found in XML.")
            return set()  # Return empty set

    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_filepath}")
        return set()
    except ET.ParseError:
        print(f"Error: Could not parse XML file at {xml_filepath}")
        return set()
    except ValueError: # Handle cases where node numbers are not integers
        print("Error: Invalid node number format found in the XML.")
        return set()





# ... (get_solid_domain_names and get_unique_node_numbers functions from previous responses)

def get_nodes_by_material(xml_filepath, material_name):
    """
    Gets all unique node numbers from parts using a given material.

    Args:
        xml_filepath: Path to the XML file.
        material_name: Material name to search for.

    Returns:
        A set of unique node numbers from all parts using the material, 
        or an empty set if none are found.
    """

    part_names = get_parts_with_material(xml_filepath, material_name)  # Get part names
    all_nodes = set()  # Set to store all unique node numbers

    if part_names:
        for part_name in part_names:
            part_nodes = get_unique_node_numbers(xml_filepath, part_name)  # Get nodes for each part
            all_nodes.update(part_nodes)  # Add nodes to the overall set

    return all_nodes


# Example usage:
# xml_file = r"your_xml_file.xml"  # Replace with your XML file path
material = "Levator Ani Material"

all_unique_nodes = get_nodes_by_material(xml_file, material)

# if all_unique_nodes:
#     print(f"All unique node numbers for material '{material}': {sorted(list(all_unique_nodes))}")
# else:
#     print(f"No node numbers found for material '{material}'.")

