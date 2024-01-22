from procoder.functional import collect_refnames, format_multiple_prompts
from procoder.functional import indent4 as indent
from procoder.prompt import *
from safeagi.prompts.principles import TOOL_CALL_VALIDITY_REQS
from safeagi.utils import get_num_tokens, make_colorful

from safeagi.prompts.globals import *

# ============= Standard Regulation rewriting =============
STD_FINETUNE_GENERATOR_INFO = """For each of the given sentences, generate 5 paraphrases with diverse text styles.
    Use xml tags to separate each paraphrase.

    For example:

    Given sentences:
    (1) No driver shall enter an intersection or crosswalk when facing a steady red signal.
    (2) Right turns on red are permitted after coming to a complete stop and ensuring the way is clear of oncoming traffic and pedestrians, unless otherwise posted.

    Rewrite each sentence in 5 different ways:
    <1-1> Under no circumstances should a driver proceed into an intersection or pedestrian crossing upon encountering a continuous red light. </1-1>
    <1-2> Hey, if you see a solid red light, don't go driving into the intersection or crosswalk! </1-2>
    <1-3> In instances where a vehicular operator encounters a non-flashing, uniform red illumination at an intersection or designated pedestrian pathway, said operator is prohibited from advancing. </1-3>
    <1-4> You know the drill: solid red light means you don't move into that intersection or crosswalk. </1-4>
    <1-5> When approaching an intersection or crosswalk, stop and remain stationary if presented with a steady red light. </1-5>
    <2-1> Turning right on a red signal is permissible, provided the vehicle has halted entirely and the driver has ascertained that there are no approaching vehicles or pedestrians. This is subject to any additional posted restrictions. </2-1>
    <2-2> So, you can make a right on red, but only if you've stopped fully and checked there's no one coming or walking by. And, of course, unless there's a sign saying otherwise. </2-2>
    <2-3> In the event of a red signal, vehicular operators are allowed to initiate a rightward maneuver contingent upon a full cessation of motion and a thorough assessment ensuring an absence of proximate vehicular and pedestrian entities, barring any explicit signage to the contrary. </2-3>
    <2-4> You know you can turn right on red, right? Just make sure you've stopped all the way and it's all clear. Oh, and watch out for any signs that say otherwise. </2-4>
    <2-5> To turn right on a red light: First, come to a full stop. Next, check for any oncoming cars or people walking. If the coast is clear and there's no sign saying you can't, then you can turn. </2-5>

    Below are the sentences to rewrite:

    Sentences:
    {}

    Rewrite each of them in 5 different text styles.
    Use the xml tags following the above example.
"""


HINDSIGHT_NEGATIVE_EXAMPLE_GENERATOR_INFO = """
Below is a user instruction and a relevant safety regulation.

User instruction:
{}

Safety regulation:
{}

Generate an action trajectory with {} that VIOLATES the safety regulation when following the user instruction and managing the task. Then explains why this action violates the regulation and provide a criticism.

Below is an example action trajectory and criticism pair, using xml tags <action> and <criticism>:

<action>
{}
</action>

<criticism>
{}
</criticism>

Follow the example above to generate a violating action trajectory with criticism.
Please use the xml tags <action> and <criticism> as in the above example.
"""


SELF_TAUGHT_EXAMPLE_GENERATOR_INFO = """Given a safety regulation, come up with a scenario that offends this principle containing:
    (1) environment
    (2) action
    (3) unwanted outcome of the action due to offending the principle
    (4) criticism over the action based on the principle

    For example:

    Safety regulation:
    Always verify the correct amount of money before sending it to anyone: Prior to completing any transaction, make sure to double-check the exact sum you are transferring. Make sure there are no additional zeros or incorrectly placed decimal points.

    <environment>
    A user needs to split a PSEG bill of $660 with 2 roommates.
    </environment>

    <action>

    </action>

    <unwanted_outcome>
    Instead of entering $50.00, Jane mistakenly enters $500.00 because of an extra zero and sends the payment. The seller receives the payment and, seeing the extra amount, decides to keep it without notifying Jane.
    </unwanted_outcome>

    <criticism>
    Had Jane adhered to the safety principle of always verifying the correct amount of money before sending it, she would have caught the mistake and saved herself $450. It's essential to always double-check transaction amounts to avoid costly errors, especially in online environments where mistakes can be easily made and harder to rectify.
    </criticism>
"""
