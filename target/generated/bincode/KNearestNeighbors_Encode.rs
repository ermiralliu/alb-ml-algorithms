impl < I, O > :: bincode :: Encode for KNearestNeighbors < I, O > where I : ::
bincode :: Encode, O : :: bincode :: Encode
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) ->core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        :: bincode :: Encode :: encode(&self.k, encoder) ?; :: bincode ::
        Encode :: encode(&self.training_data, encoder) ?; :: bincode :: Encode
        :: encode(&self.training_labels, encoder) ?; core :: result :: Result
        :: Ok(())
    }
}