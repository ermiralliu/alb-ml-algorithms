impl :: bincode :: Encode for NaiveBayes
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) ->core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        :: bincode :: Encode :: encode(&self.class_log_priors, encoder) ?; ::
        bincode :: Encode :: encode(&self.class_log_likelihoods, encoder) ?;
        :: bincode :: Encode :: encode(&self.vocabulary_size, encoder) ?; core
        :: result :: Result :: Ok(())
    }
}