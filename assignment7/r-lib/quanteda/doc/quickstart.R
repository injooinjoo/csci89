## ----echo = FALSE-------------------------------------------------------------
knitr::opts_chunk$set(collapse = FALSE, comment = "##")

## ----eval = FALSE-------------------------------------------------------------
# install.packages("quanteda")

## ----eval = FALSE-------------------------------------------------------------
# remotes::install_github("quanteda/quanteda.corpora")

## ----eval = FALSE-------------------------------------------------------------
# remotes::install_github("kbenoit/quanteda.dictionaries")

## ----message = FALSE----------------------------------------------------------
library("quanteda")

## ----include=FALSE------------------------------------------------------------
quanteda_options(threads = 1)

## -----------------------------------------------------------------------------
corp_uk <- corpus(data_char_ukimmig2010)  # build a new corpus from the texts
summary(corp_uk)

## -----------------------------------------------------------------------------
docvars(corp_uk, "Party") <- names(data_char_ukimmig2010)
docvars(corp_uk, "Year") <- 2010
summary(corp_uk)

## ----eval=FALSE---------------------------------------------------------------
# require(readtext)
# 
# # Twitter json
# dat_json <- readtext("social_media/zombies/tweets.json")
# corp_twitter <- corpus(dat_json)
# summary(corp_twitter, 5)
# 
# # generic json - needs a textfield specifier
# dat_sotu <- readtext("corpora/sotu/sotu.json", text_field = "text")
# summary(corpus(dat_sotu), 5)
# 
# # text file
# dat_txtone <- readtext("corpora/project_gutenberg/pg2701.txt")
# summary(corpus(dat_txtone), 5)
# 
# # multiple text files
# dat_txtmultiple1 <- readtext("corpora/inaugural/*.txt")
# summary(corpus(dat_txtmultiple1), 5)
# 
# # multiple text files with docvars from filenames
# dat_txtmultiple2 <- readtext("corpora/inaugural/*.txt",
#                              docvarsfrom = "filenames", sep = "-",
#                              docvarnames = c("Year", "President"))
# summary(corpus(dat_txtmultiple2), 5)
# 
# # XML data
# dat_xml <- readtext("xmlData/plant_catalog.xml", text_field = "COMMON")
# summary(corpus(dat_xml), 5)
# 
# # csv file
# write.csv(data.frame(inaug_speech = as.character(data_corpus_inaugural),
#                      docvars(data_corpus_inaugural)),
#           file = "/tmp/inaug_texts.csv", row.names = FALSE)
# dat_csv <- readtext("/tmp/inaug_texts.csv", text_field = "inaug_speech")
# summary(corpus(dat_csv), 5)

## -----------------------------------------------------------------------------
print(data_corpus_inaugural)

## -----------------------------------------------------------------------------
as.character(data_corpus_inaugural)[2]

## -----------------------------------------------------------------------------
summary(data_corpus_inaugural, n = 5)

## ----fig.width = 8------------------------------------------------------------
tokeninfo <- summary(data_corpus_inaugural)
tokeninfo$Year <- docvars(data_corpus_inaugural, "Year")
with(tokeninfo, plot(Year, Tokens, type = "b", pch = 19, cex = .7))

## -----------------------------------------------------------------------------
# longest inaugural address: William Henry Harrison
tokeninfo[which.max(tokeninfo$Tokens), ]

## -----------------------------------------------------------------------------
corp1 <- head(data_corpus_inaugural, 2)
corp2 <- tail(data_corpus_inaugural, 2)
corp3 <- corp1 + corp2
summary(corp3)

## -----------------------------------------------------------------------------
summary(corpus_subset(data_corpus_inaugural, Year > 1990))
summary(corpus_subset(data_corpus_inaugural, President == "Adams"))

## -----------------------------------------------------------------------------
data_tokens_inaugural <- tokens(data_corpus_inaugural)
kwic(data_tokens_inaugural, pattern = "terror")

## -----------------------------------------------------------------------------
kwic(data_tokens_inaugural, pattern = "terror", valuetype = "regex")

## -----------------------------------------------------------------------------
kwic(data_tokens_inaugural, pattern = "communist*")

## -----------------------------------------------------------------------------
# show context of the first six occurrences of "United States"
kwic(data_tokens_inaugural, pattern = phrase("United States")) |>
    head()

## -----------------------------------------------------------------------------
# inspect the document-level variables
head(docvars(data_corpus_inaugural))

## -----------------------------------------------------------------------------
txt <- c(text1 = "This is $10 in 999 different ways,\n up and down; left and right!",
         text2 = "@koheiw7 working: on #quanteda 2day\t4ever, http://textasdata.com?page=123.")
tokens(txt)
tokens(txt, remove_numbers = TRUE,  remove_punct = TRUE)
tokens(txt, remove_numbers = FALSE, remove_punct = TRUE)
tokens(txt, remove_numbers = TRUE,  remove_punct = FALSE)
tokens(txt, remove_numbers = FALSE, remove_punct = FALSE)
tokens(txt, remove_numbers = FALSE, remove_punct = FALSE, remove_separators = FALSE)

## -----------------------------------------------------------------------------
tokens("Great website: http://textasdata.com?page=123.", what = "character")
tokens("Great website: http://textasdata.com?page=123.", what = "character",
         remove_separators = FALSE)

## -----------------------------------------------------------------------------
# sentence level       
tokens(c("Kurt Vongeut said; only assholes use semi-colons.",
         "Today is Thursday in Canberra:  It is yesterday in London.",
         "En el caso de que no puedas ir con ellos, ¿quieres ir con nosotros?"),
          what = "sentence")

## -----------------------------------------------------------------------------
head(stopwords("en"), 20)
head(stopwords("ru"), 10)
head(stopwords("ar", source = "misc"), 10)

## -----------------------------------------------------------------------------
tokens("New York City is located in the United States.") |>
    tokens_compound(pattern = phrase(c("New York City", "United States")))

## -----------------------------------------------------------------------------
tokens("one~two~three") |>
    tokens_split(separator = "~")

## -----------------------------------------------------------------------------
corp_inaug_post1990 <- corpus_subset(data_corpus_inaugural, Year > 1990)

# make a dfm
dfmat_inaug_post1990 <- corp_inaug_post1990 |>
    tokens() |>
    dfm()
print(dfmat_inaug_post1990)

## ----fig.width = 8, fig.height = 8--------------------------------------------
dfmat_uk <- tokens(data_char_ukimmig2010, remove_punct = TRUE) |>
  tokens_remove(stopwords("en")) |>
  dfm()
dfmat_uk

## -----------------------------------------------------------------------------
# 20 most frequent words
topfeatures(dfmat_uk, 20)

## -----------------------------------------------------------------------------
dfmat_pres <- tail(data_corpus_inaugural, 20) |>
  tokens(remove_punct = TRUE) |>
  tokens_remove(stopwords("en")) |>
  dfm() |>
  dfm_group(groups = Party)

## -----------------------------------------------------------------------------
dfm_sort(dfmat_pres)

## -----------------------------------------------------------------------------
corp_inaug_post1991 <- corpus_subset(data_corpus_inaugural, Year > 1991)

## -----------------------------------------------------------------------------
dict <- dictionary(list(terror = c("terrorism", "terrorists", "threat"),
                        economy = c("jobs", "business", "grow", "work")))

## -----------------------------------------------------------------------------
dfmat_inaug_post1991_dict <- tokens(corp_inaug_post1991) |>
  tokens_lookup(dictionary = dict) |>
  dfm()
dfmat_inaug_post1991_dict

## ----eval = FALSE-------------------------------------------------------------
# dictliwc <- dictionary(file = "LIWC2001_English.dic", format = "LIWC")
# dfmat_inaug_subset <- tokens(data_corpus_inaugural[52:58]) |>
#                       dfm() |>
#                       dfm_lookup(dictionary = dictliwc)
# dfmat_inaug_subset[, 1:10]

