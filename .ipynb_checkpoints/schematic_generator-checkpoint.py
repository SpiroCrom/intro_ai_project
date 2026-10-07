import random
import os
import schemdraw
import schemdraw.elements as elm


# Folder where the generated circuit images will be saved
OUTPUT_FOLDER = "generated_circuits"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def random_resistor():
    """Return a random resistor value."""
    values = ["1kΩ", "2kΩ", "4.7kΩ", "5kΩ", "10kΩ", "22kΩ"]
    return random.choice(values)


def random_voltage():
    """Return a random voltage source value."""
    values = ["5V", "9V", "10V", "12V", "15V"]
    return random.choice(values)


def draw_circuit(circuit_id):

    # Randomly choose ONE main circuit concept
    circuit_type = random.choice([
        "standard",
        "voltage_divider",
        "current_divider",
        "supernode"
    ])

    labels = {
        "has_supernode": False,
        "has_voltage_divider": False,
        "has_current_divider": False
    }

    d = schemdraw.Drawing(show=False)

    # -----------------------------------
    # STANDARD CIRCUIT
    # -----------------------------------
    if circuit_type == "standard":

        voltage = random_voltage()
        resistance = random_resistor()

        d += elm.SourceV().up().label(f"V1\n{voltage}")
        d += elm.Line().right()
        d += elm.Resistor().down().label(f"R1\n{resistance}")
        d += elm.Line().left()

    # -----------------------------------
    # VOLTAGE DIVIDER
    # -----------------------------------
    elif circuit_type == "voltage_divider":

        labels["has_voltage_divider"] = True

        voltage = random_voltage()
        r1 = random_resistor()
        r2 = random_resistor()

        d += elm.SourceV().up().label(f"V1\n{voltage}")

        d += elm.Line().right()
        d += elm.Dot().label("Node 1", loc="top")

        d += elm.Resistor().down().label(f"R1\n{r1}")

        d += elm.Dot().label("Vout", loc="right")

        d += elm.Resistor().down().label(f"R2\n{r2}")

        d += elm.Line().left()

    # -----------------------------------
    # CURRENT DIVIDER
    # -----------------------------------
    elif circuit_type == "current_divider":

        labels["has_current_divider"] = True

        voltage = random_voltage()
        r1 = random_resistor()
        r2 = random_resistor()

        # Source
        source = d.add(elm.SourceV().up().label(f"V1\n{voltage}"))

        # Top wire
        d += elm.Line().right()

        # Save top node position
        top_node = d.here

        d += elm.Dot().label("Node 1", loc="top")

        # First parallel resistor
        d += elm.Resistor().down().label(f"R1\n{r1}")
        bottom_node = d.here

        # Bottom wire back toward source
        d += elm.Line().left().tox(source.start)

        # Return to top node
        d.move_from(top_node)

        # Move right to create second branch
        d += elm.Line().right()

        # Second resistor
        d += elm.Resistor().down().label(f"R2\n{r2}")

        # Connect R2 bottom to R1 bottom
        d += elm.Line().left().to(bottom_node)

    # -----------------------------------
    # SUPERNODE
    # -----------------------------------
    elif circuit_type == "supernode":

        labels["has_supernode"] = True

        source_voltage = random_voltage()
        bridge_voltage = random_voltage()

        r1 = random_resistor()
        r2 = random_resistor()

        # Ground reference
        d += elm.Ground()

        # Resistor from ground to Node 1
        d += elm.Resistor().up().label(f"R1\n{r1}")

        d += elm.Dot().label("Node 1", loc="left")

        # Voltage source BETWEEN two non-reference nodes
        d += elm.SourceV().right().label(
            f"V2\n{bridge_voltage}"
        )

        d += elm.Dot().label("Node 2", loc="right")

        # Resistor from Node 2
        d += elm.Resistor().down().label(f"R2\n{r2}")

        d += elm.Ground()

    # -----------------------------------
    # SAVE IMAGE
    # -----------------------------------

    filename = os.path.join(
        OUTPUT_FOLDER,
        f"circuit_{circuit_id}_{circuit_type}.png"
    )

    d.save(filename, dpi=300)

    print(f"\nCircuit #{circuit_id}")
    print(f"Type: {circuit_type}")
    print(f"Supernode: {labels['has_supernode']}")
    print(f"Voltage Divider: {labels['has_voltage_divider']}")
    print(f"Current Divider: {labels['has_current_divider']}")
    print(f"Saved as: {filename}")

    return labels


# ===================================
# GENERATE EXACTLY 5 CIRCUIT IMAGES
# ===================================

dataset_labels = []

for i in range(1, 6):

    labels = draw_circuit(i)

    dataset_labels.append({
        "image": i,
        **labels
    })


# Display dataset labels
print("\n----------------------------")
print("DATASET LABELS")
print("----------------------------")

for item in dataset_labels:
    print(item)

print("Images saved here:")
print(os.path.abspath(OUTPUT_FOLDER))