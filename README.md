# AI-Powered Content Taxonomy

Create Sphinx `Meta` directive block in CSV and add it to your RST pages. Helps documentation teams in mapping deployment type, topics, and user roles to web pages based on the input provided. Generate SEO technical keywords and map them to your web pages. 

## Map and Generate Keywords

1. In the [AI-Powered Content Taxonomy](https://sudodocs-content-mapper.streamlit.app/) application, navigate to  **Step 1: Ingest and Map** \> **Scrape URLs and Content** tab.  

2. **Browse** and upload the assigned TXT file (containing URLs) that you unzipped and then click **Scrape URLs.**  

3. In the **Map User Roles, Topics, and Functional Areas** section:  

   1. In the **Upload User Roles (.txt)** section, browse the `user_roles.txt` file and click **Map Roles**.  

   2. In the **Upload Topics (.txt)** section, browse the `topics.txt` file and click **Map Topics**.  

   3. In the **Upload Areas (.txt)** section, browse the `functional_area.txt` file and click **Map Functional Areas**.  

4. Proceed to the **Step 2: Generate Keywords** tab and in the left navigation pane, select **Choose AI Provider** as `Google Gemini` and copy-paste your Gemini API keys.   

5. Click **Fill Blanks and Generate Keywords** and wait till the application processes.  

6. Navigate to the **Step 3: Edit & Download Results** tab, verify your taxonomy data in the **Interactive Results Editor**.  

7. (Optional) Make edits if required and download the CSV file.

8. (Recommended) Return to the **Step 2: Generate Keywords** tab and use the Refine with Uniqueness Analysis section to further refine your data using AI.  

   1. in the left navigation pane, again select **Choose AI Provider** as `Google Gemini` and copy-paste your Gemini API keys provided before.  

   2. Click **Analyze and Refine Uniqueness** to generate the refined CSV data.  

9. Navigate to the **Step 3: Edit & Download Results** tab, verify your taxonomy data in the **Interactive Results Editor**.  

10. (Optional) Make edits if required and download the refined CSV file.

11.  Use the [final script](https://github.com/mrsauravs/alation_scripts/blob/main/update_rst_metadata.py) to inject `meta` directive block at the top of the RST pages.
