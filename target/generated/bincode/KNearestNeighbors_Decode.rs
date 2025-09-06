impl < I, O, __Context > :: bincode :: Decode < __Context > for
KNearestNeighbors < I, O > where I : :: bincode :: Decode < __Context > , O :
:: bincode :: Decode < __Context >
{
    fn decode < __D : :: bincode :: de :: Decoder < Context = __Context > >
    (decoder : & mut __D) ->core :: result :: Result < Self, :: bincode ::
    error :: DecodeError >
    {
        core :: result :: Result ::
        Ok(Self
        {
            k : :: bincode :: Decode :: decode(decoder) ?, training_data : ::
            bincode :: Decode :: decode(decoder) ?, training_labels : ::
            bincode :: Decode :: decode(decoder) ?,
        })
    }
} impl < '__de, I, O, __Context > :: bincode :: BorrowDecode < '__de,
__Context > for KNearestNeighbors < I, O > where I : :: bincode :: de ::
BorrowDecode < '__de, __Context > , O : :: bincode :: de :: BorrowDecode <
'__de, __Context >
{
    fn borrow_decode < __D : :: bincode :: de :: BorrowDecoder < '__de,
    Context = __Context > > (decoder : & mut __D) ->core :: result :: Result <
    Self, :: bincode :: error :: DecodeError >
    {
        core :: result :: Result ::
        Ok(Self
        {
            k : :: bincode :: BorrowDecode ::< '_, __Context >::
            borrow_decode(decoder) ?, training_data : :: bincode ::
            BorrowDecode ::< '_, __Context >:: borrow_decode(decoder) ?,
            training_labels : :: bincode :: BorrowDecode ::< '_, __Context >::
            borrow_decode(decoder) ?,
        })
    }
}