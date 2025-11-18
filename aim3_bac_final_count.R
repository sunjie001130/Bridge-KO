library(dplyr)

# Path where your 10 CSVs are saved
dir_path <- "/ix/hpark/Jie/LMM/Bac_final_model/"   # change this

# List all CSV files (assuming pattern like selected_features_run1.csv, etc.)
files <- list.files(dir_path, pattern = "selected_features_run.*\\.csv$", full.names = TRUE)

# Read and combine all selected features
all_selected <- lapply(files, function(f) {
  df <- read.csv(f)
  df$run <- basename(f)  # keep which run this came from
  df
}) %>% bind_rows()

# Count how many times each feature was selected
feature_counts <- all_selected %>%
  group_by(feature) %>%
  summarise(selection_count = n()) %>%
  arrange(desc(selection_count))

# Preview
head(feature_counts)

# RA Genus from aim2
feature_counts <- feature_counts %>%
  mutate(RA = ifelse(feature %in% RA_Genus, "yes", "no"))

# Save to a new CSV
write.csv(feature_counts, file.path(dir_path, "selected_feature_summary.csv"), row.names = FALSE)