import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class RetailAnalyzer:

    def __init__(self):
        self.data = None

    def load_data(self, file_path):

        self.data = pd.read_csv(file_path, sep=None, engine="python")

        required_columns = [
            "Date",
            "Product",
            "Category",
            "Price",
            "Quantity Sold",
            "Total Sales"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {', '.join(missing_columns)}"
            )

        self.data["Date"] = pd.to_datetime(
            self.data["Date"],
            errors="coerce"
        )

        numeric_columns = [
            "Price",
            "Quantity Sold",
            "Total Sales"
        ]

        for column in numeric_columns:
            self.data[column] = pd.to_numeric(
                self.data[column],
                errors="coerce"
            )

        self.data.dropna(
            subset=[
                "Date",
                "Price",
                "Quantity Sold",
                "Total Sales"
            ],
            inplace=True
        )

        print("\nCSV file loaded successfully!")

    def calculate_metrics(self):

        total_sales = self.data["Total Sales"].sum()
        average_sales = self.data["Total Sales"].mean()

        product_sales = self.data.groupby(
            "Product"
        )["Quantity Sold"].sum()

        print("\n----- SALES METRICS -----")
        print("Total Sales:", round(total_sales, 2))
        print("Average Sales:", round(average_sales, 2))

        if not product_sales.empty:
            print(
                "Most Popular Product:",
                product_sales.idxmax()
            )

    def filter_data(self, category):

        return self.data[
            self.data["Category"].astype(str).str.lower()
            == category.strip().lower()
        ]

    def display_summary(self):

        print("\n----- DATA SUMMARY -----")
        print("Total Records:", len(self.data))
        print("Total Columns:", len(self.data.columns))

        print("\nColumn Names:")
        print(self.data.columns.tolist())

        print("\nMissing Values:")
        print(self.data.isnull().sum())


print("===================================")
print("      RETAIL SALES DATA ANALYZER")
print("===================================")

file_path = input(
    "\nEnter CSV file name or path: "
).strip()

analyzer = RetailAnalyzer()

try:
    analyzer.load_data(file_path)

except FileNotFoundError:
    print("\nError: CSV file not found.")
    raise SystemExit

except Exception as error:
    print("\nError:", error)
    raise SystemExit


df = analyzer.data


print("\n----- FIRST 5 RECORDS -----")
print(df.head())


analyzer.display_summary()


print("\n----- CHECKING QUANTITY -----")

invalid_quantity = df[df["Quantity Sold"] <= 0]

if invalid_quantity.empty:
    print("All quantities are valid.")

else:
    print(
        invalid_quantity[
            ["Product", "Quantity Sold"]
        ]
    )


analyzer.calculate_metrics()


category = input(
    "\nEnter category to filter: "
).strip()

filtered_data = analyzer.filter_data(category)


print("\n----- FILTERED DATA -----")

if filtered_data.empty:
    print("No data found for this category.")

else:
    print(filtered_data)


print("\n----- DATE FILTER -----")

start_date = pd.to_datetime(
    input("Enter start date (YYYY-MM-DD): ").strip(),
    errors="coerce"
)

end_date = pd.to_datetime(
    input("Enter end date (YYYY-MM-DD): ").strip(),
    errors="coerce"
)


if pd.isna(start_date) or pd.isna(end_date):

    print("Invalid date format. Use YYYY-MM-DD.")

else:

    date_data = df[
        (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    ]

    print("\n----- DATE FILTERED DATA -----")

    if date_data.empty:
        print("No data found for this date range.")

    else:
        print(date_data)


sales = df["Total Sales"].to_numpy()


print("\n----- NUMPY RESULTS -----")
print("Highest Sales:", np.max(sales))
print("Lowest Sales:", np.min(sales))
print("Average Sales:", round(np.mean(sales), 2))


category_sales = df.groupby(
    "Category"
)["Total Sales"].sum()


print("\n----- SALES BY CATEGORY -----")
print(category_sales)


product_sales = df.groupby(
    "Product"
)["Total Sales"].sum()


print("\n----- SALES BY PRODUCT -----")
print(product_sales)


df["Sales Per Unit"] = np.where(
    df["Quantity Sold"] > 0,
    df["Total Sales"] / df["Quantity Sold"],
    np.nan
)


print("\n----- SALES PER UNIT -----")

print(
    df[
        [
            "Product",
            "Quantity Sold",
            "Total Sales",
            "Sales Per Unit"
        ]
    ]
)


first_sale = df["Total Sales"].iloc[0]
last_sale = df["Total Sales"].iloc[-1]


if first_sale != 0:

    growth = (
        (last_sale - first_sale)
        / first_sale
    ) * 100

    print(
        "\nSales Growth:",
        round(growth, 2),
        "%"
    )

else:

    print(
        "\nSales Growth: "
        "Cannot calculate because first sale is zero."
    )


plt.figure(figsize=(7, 5))

plt.bar(
    category_sales.index,
    category_sales.values
)

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


date_sales = df.groupby(
    "Date"
)["Total Sales"].sum()


plt.figure(figsize=(9, 5))

plt.plot(
    date_sales.index,
    date_sales.values,
    marker="o"
)

plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


correlation = df[
    [
        "Price",
        "Quantity Sold",
        "Total Sales"
    ]
].corr()


plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Sales Correlation")

plt.tight_layout()
plt.show()


print("\n===================================")
print("          FINAL SUMMARY")
print("===================================")

print(
    "Total Sales:",
    round(df["Total Sales"].sum(), 2)
)

print(
    "Average Sales:",
    round(df["Total Sales"].mean(), 2)
)


popular_product = df.groupby(
    "Product"
)["Quantity Sold"].sum().idxmax()


best_category = df.groupby(
    "Category"
)["Total Sales"].sum().idxmax()


print(
    "Most Popular Product:",
    popular_product
)

print(
    "Best Category:",
    best_category
)


print("\nProject completed successfully!")