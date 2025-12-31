import pandas as pd

from arxiv_rec.data.clean import prepare_corpus


def test_prepare_corpus_combines_text_and_filters_empty():
    df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "title": ["  Title  ", ""],
            "abstract": ["  Abstract ", ""],
            "categories": ["cs.AI", "cs.LG"],
        }
    )

    cleaned = prepare_corpus(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["text"] == "Title. Abstract"
