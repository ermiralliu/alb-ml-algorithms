impl < __Context > :: bincode :: Decode < __Context > for NaiveBayes
{
    fn decode < __D : :: bincode :: de :: Decoder < Context = __Context > >
    (decoder : & mut __D) ->core :: result :: Result < Self, :: bincode ::
    error :: DecodeError >
    {
        core :: result :: Result ::
        Ok(Self
        {
            class_log_priors : :: bincode :: Decode :: decode(decoder) ?,
            class_log_likelihoods : :: bincode :: Decode :: decode(decoder) ?,
            doc_counts : :: bincode :: Decode :: decode(decoder) ?,
            word_counts : :: bincode :: Decode :: decode(decoder) ?,
            vocabulary_size : :: bincode :: Decode :: decode(decoder) ?,
            class_word_frequencies : :: bincode :: Decode :: decode(decoder)
            ?, total_docs_processed : :: bincode :: Decode :: decode(decoder)
            ?,
        })
    }
} impl < '__de, __Context > :: bincode :: BorrowDecode < '__de, __Context >
for NaiveBayes
{
    fn borrow_decode < __D : :: bincode :: de :: BorrowDecoder < '__de,
    Context = __Context > > (decoder : & mut __D) ->core :: result :: Result <
    Self, :: bincode :: error :: DecodeError >
    {
        core :: result :: Result ::
        Ok(Self
        {
            class_log_priors : :: bincode :: BorrowDecode ::< '_, __Context
            >:: borrow_decode(decoder) ?, class_log_likelihoods : :: bincode
            :: BorrowDecode ::< '_, __Context >:: borrow_decode(decoder) ?,
            doc_counts : :: bincode :: BorrowDecode ::< '_, __Context >::
            borrow_decode(decoder) ?, word_counts : :: bincode :: BorrowDecode
            ::< '_, __Context >:: borrow_decode(decoder) ?, vocabulary_size :
            :: bincode :: BorrowDecode ::< '_, __Context >::
            borrow_decode(decoder) ?, class_word_frequencies : :: bincode ::
            BorrowDecode ::< '_, __Context >:: borrow_decode(decoder) ?,
            total_docs_processed : :: bincode :: BorrowDecode ::< '_,
            __Context >:: borrow_decode(decoder) ?,
        })
    }
}