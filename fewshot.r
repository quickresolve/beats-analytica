# =============================================================================
# Few-Shot Genre Classification: Claude LLM vs Random Forest
#
# Loads high/low-popularity Spotify datasets, builds few-shot examples from
# the training split, sends batched classification requests to the Claude API,
# and compares accuracy against a Random Forest baseline.
#
# Requirements:
#   install.packages(c("httr2", "jsonlite", "randomForest"))
#   Sys.setenv(ANTHROPIC_API_KEY = "<your key>")
# =============================================================================

# ── 0. Packages ───────────────────────────────────────────────────────────────
for (pkg in c("httr2", "jsonlite", "randomForest")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# ── 1. Load & combine datasets ────────────────────────────────────────────────
FEATURES <- c("energy", "tempo", "danceability", "loudness", "liveness",
              "valence", "speechiness", "instrumentalness", "acousticness",
              "key", "mode", "time_signature", "duration_ms")
TARGET <- "playlist_genre"

high <- read.csv("spotify-music-dataset/high_popularity_spotify_data.csv",
                 stringsAsFactors = FALSE)
low  <- read.csv("spotify-music-dataset/low_popularity_spotify_data.csv",
                 stringsAsFactors = FALSE)

data <- rbind(high[, c(FEATURES, TARGET)],
              low[,  c(FEATURES, TARGET)])
data <- na.omit(data)
data[[TARGET]] <- trimws(tolower(data[[TARGET]]))

# ── 2. Keep genres with >= 30 samples ────────────────────────────────────────
genre_counts <- sort(table(data[[TARGET]]), decreasing = TRUE)
cat("All genres and counts:\n")
print(genre_counts)

keep_genres <- names(genre_counts[genre_counts >= 30])
data <- data[data[[TARGET]] %in% keep_genres, ]
data[[TARGET]] <- factor(data[[TARGET]])
cat(sprintf(
  "\nUsing %d genres (>= 30 samples each): %s\n",
  length(keep_genres), paste(sort(keep_genres), collapse = ", ")
))

# ── 3. Stratified 80/20 train/test split ─────────────────────────────────────
set.seed(42)
train_idx <- c()
test_idx  <- c()
for (g in keep_genres) {
  idx     <- which(data[[TARGET]] == g)
  n_train <- max(5, round(length(idx) * 0.8))   # ≥5 so few-shot has candidates
  sel     <- sample(idx, n_train)
  train_idx <- c(train_idx, sel)
  test_idx  <- c(test_idx, setdiff(idx, sel))
}
train_data <- data[train_idx, ]
test_data  <- data[test_idx,  ]
cat(sprintf("Train: %d rows  |  Test: %d rows\n\n", nrow(train_data), nrow(test_data)))

# ── 4. Random Forest baseline ─────────────────────────────────────────────────
cat("Training Random Forest (ntree=300)...\n")
rf_model <- randomForest(
  x         = train_data[, FEATURES],
  y         = train_data[[TARGET]],
  ntree     = 300,
  importance = TRUE
)
rf_preds_full <- predict(rf_model, test_data[, FEATURES])
rf_acc_full   <- mean(rf_preds_full == test_data[[TARGET]])
cat(sprintf("RF accuracy (full test set, n=%d): %.1f%%\n\n",
            nrow(test_data), rf_acc_full * 100))

# ── 5. Few-shot prompt construction ──────────────────────────────────────────
N_SHOT <- 3   # examples per genre

# Compact single-line track representation
fmt_track <- function(row) {
  sprintf(
    "energy=%.2f tempo=%.0f dance=%.2f loud=%.1fdB live=%.2f valence=%.2f "   ,
    row$energy, row$tempo, row$danceability, row$loudness, row$liveness, row$valence
  ) |> paste0(sprintf(
    "speech=%.3f instr=%.3f acoustic=%.2f key=%d mode=%d tsig=%d dur=%ds",
    row$speechiness, row$instrumentalness, row$acousticness,
    row$key, row$mode, row$time_signature,
    round(row$duration_ms / 1000)
  ))
}

few_shot_lines <- character(0)
for (g in sort(keep_genres)) {
  rows  <- train_data[train_data[[TARGET]] == g, ]
  n_use <- min(N_SHOT, nrow(rows))
  samp  <- rows[sample(nrow(rows), n_use), ]
  for (i in seq_len(nrow(samp))) {
    few_shot_lines <- c(few_shot_lines,
      paste0("[", fmt_track(samp[i, ]), "] → ", g)
    )
  }
}
few_shot_block <- paste(few_shot_lines, collapse = "\n")

# ── 6. Claude API helper ──────────────────────────────────────────────────────
call_claude <- function(prompt,
                        model      = "claude-opus-4-6",
                        max_tokens = 256) {
  api_key <- Sys.getenv("ANTHROPIC_API_KEY")
  if (!nzchar(api_key))
    stop("Set ANTHROPIC_API_KEY environment variable before running.")

  resp <- request("https://api.anthropic.com/v1/messages") |>
    req_headers(
      "x-api-key"         = api_key,
      "anthropic-version" = "2023-06-01",
      "content-type"      = "application/json"
    ) |>
    req_body_json(list(
      model      = model,
      max_tokens = max_tokens,
      messages   = list(list(role = "user", content = prompt))
    )) |>
    req_error(is_error = \(r) FALSE) |>
    req_perform()

  if (resp_status(resp) != 200)
    stop(sprintf("API error %d: %s", resp_status(resp), resp_body_string(resp)))

  resp_body_json(resp)$content[[1]]$text
}

# ── 7. Batch LLM classifier ───────────────────────────────────────────────────
BATCH_SIZE   <- 5      # tracks per API call
MAX_TEST_N   <- 100    # cap for cost control (set NULL for full test set)

# Sub-sample test set if capped
set.seed(99)
if (!is.null(MAX_TEST_N) && nrow(test_data) > MAX_TEST_N) {
  # Stratified subsample to keep genre balance
  sub_idx <- c()
  per_g   <- max(1, floor(MAX_TEST_N / length(keep_genres)))
  for (g in keep_genres) {
    g_idx <- which(test_data[[TARGET]] == g)
    n_use <- min(per_g, length(g_idx))
    sub_idx <- c(sub_idx, sample(g_idx, n_use))
  }
  test_sub <- test_data[sub_idx, ]
} else {
  test_sub <- test_data
}
cat(sprintf("LLM classification on %d test samples (%d genres × ~%d each)...\n",
            nrow(test_sub), length(keep_genres),
            round(nrow(test_sub) / length(keep_genres))))

classify_batch <- function(rows, few_shot_block, genres) {
  genre_str  <- paste(sort(genres), collapse = ", ")
  track_strs <- vapply(seq_len(nrow(rows)), function(i)
    sprintf("%d. [%s]", i, fmt_track(rows[i, ])), character(1))

  prompt <- paste0(
    "You are a music genre classifier. Predict each track's genre from its audio features.\n\n",
    "Valid genres: ", genre_str, "\n\n",
    "Examples (Features → Genre):\n", few_shot_block, "\n\n",
    "Classify each track below. Reply with ONLY a numbered list where each line is ",
    "exactly: <number>. <genre>  (one line per track, no extra text):\n\n",
    paste(track_strs, collapse = "\n")
  )

  raw <- call_claude(prompt, max_tokens = BATCH_SIZE * 12 + 50)

  # Parse: strip numbering, lowercase, map to known genre
  lines   <- strsplit(trimws(raw), "\n")[[1]]
  lines   <- lines[nchar(trimws(lines)) > 0]
  parsed  <- gsub("^[0-9]+[.):\\s]+\\s*", "", lines)
  parsed  <- trimws(tolower(parsed))

  genres_lower <- tolower(genres)
  matched <- vapply(parsed, function(p) {
    # exact match first
    m <- match(p, genres_lower)
    if (!is.na(m)) return(genres[m])
    # partial: first genre that appears as substring
    hits <- which(sapply(genres_lower, function(g) grepl(g, p, fixed = TRUE)))
    if (length(hits) > 0) return(genres[hits[1]])
    NA_character_
  }, character(1))

  length(matched) <- nrow(rows)   # pad/truncate to batch size
  matched
}

# Run batched classification
true_labels <- as.character(test_sub[[TARGET]])
llm_preds   <- character(nrow(test_sub))
batches     <- split(seq_len(nrow(test_sub)),
                     ceiling(seq_len(nrow(test_sub)) / BATCH_SIZE))

cat(sprintf("Sending %d batches of up to %d tracks to Claude (%s)...\n\n",
            length(batches), BATCH_SIZE, "claude-opus-4-6"))

for (b in seq_along(batches)) {
  idx      <- batches[[b]]
  batch_df <- test_sub[idx, ]

  preds <- tryCatch(
    classify_batch(batch_df, few_shot_block, keep_genres),
    error = function(e) {
      message(sprintf("\n  Batch %d/%d failed: %s", b, length(batches),
                      conditionMessage(e)))
      rep(NA_character_, length(idx))
    }
  )

  if (length(preds) < length(idx))
    preds <- c(preds, rep(NA_character_, length(idx) - length(preds)))
  llm_preds[idx] <- preds[seq_along(idx)]

  cat(sprintf("  Batch %2d/%d done — genres: %s\n",
              b, length(batches), paste(preds, collapse = ", ")))
  Sys.sleep(0.4)   # gentle rate limiting
}

# ── 8. Results ────────────────────────────────────────────────────────────────
valid       <- !is.na(llm_preds)
llm_acc     <- mean(llm_preds[valid] == true_labels[valid])
rf_preds_sub <- as.character(predict(rf_model, test_sub[, FEATURES]))
rf_acc_sub   <- mean(rf_preds_sub == true_labels)

cat(sprintf("\n%s\n", strrep("=", 60)))
cat(sprintf("RESULTS  (test n=%d, %d/%d parseable LLM responses)\n",
            nrow(test_sub), sum(valid), nrow(test_sub)))
cat(sprintf("%s\n", strrep("=", 60)))
cat(sprintf("  LLM few-shot (claude-opus-4-6) : %5.1f%%\n", llm_acc   * 100))
cat(sprintf("  Random Forest (ntree=300)       : %5.1f%%\n", rf_acc_sub * 100))
cat(sprintf("  Lift (LLM - RF)                 : %+.1f pp\n\n",
            (llm_acc - rf_acc_sub) * 100))

# Per-genre breakdown
cat(sprintf("%-16s  %7s  %7s  %5s\n", "Genre", "LLM", "RF", "n"))
cat(strrep("-", 42), "\n")
for (g in sort(keep_genres)) {
  g_all   <- which(true_labels == g)
  g_valid <- which(true_labels == g & valid)
  if (length(g_all) == 0) next
  llm_g   <- if (length(g_valid) > 0) mean(llm_preds[g_valid] == g) else NA
  rf_g    <- mean(rf_preds_sub[g_all] == g)
  cat(sprintf("%-16s  %6.1f%%  %6.1f%%  %5d\n",
              g,
              if (is.na(llm_g)) NA else llm_g * 100,
              rf_g * 100,
              length(g_all)))
}

# Confusion matrix for LLM predictions
cat(sprintf("\n%s\n", strrep("-", 60)))
cat("LLM Confusion Matrix (rows = Predicted, cols = Actual):\n")
genres_present <- sort(unique(true_labels))
cm <- table(
  Predicted = factor(llm_preds[valid], levels = genres_present),
  Actual    = factor(true_labels[valid], levels = genres_present)
)
print(cm)

# Random Forest variable importance
cat(sprintf("\n%s\n", strrep("-", 60)))
cat("Top 7 features by RF Mean Decrease Gini:\n")
imp   <- importance(rf_model, type = 2)
top_n <- head(sort(imp[, "MeanDecreaseGini"], decreasing = TRUE), 7)
for (nm in names(top_n))
  cat(sprintf("  %-18s %.2f\n", nm, top_n[nm]))
