# interactive_gradients
An Interactive Human-Machine Learning Interface for Collecting and Learning from Complex Annotations

Welcome to the code repo for the interactive gradients tool described in our paper:
- preprint: https://arxiv.org/pdf/2403.19339
- IJCAI Demo: [I will add a link to the IJCAI demo paper once it has been published].

To run the code, install the necessary libraries:
    `pip install -r requirements.txt`

Then run the gradient_supervision script:
`python gradient_supervision.py`

Any results you saved will be pickled and stored in `interactive_results/default_session.pkl`. You can add to these files by changing the `session_id` variable [Presently at: gradient_supervision.py - line 62]

Version 2 will hopefully be available soon and will experiment with similar ideas in the language domain!

If you have any questions feel free to leave comments/requests.