# Task 2 comparison - `custom_20`

Protocol A (greedy / zero temperature), first sentence of each continuation. Generated locally on Apple MPS.

## Factual

| Prompt | Expected | ours: TinyStories-1M | ours: TinyStories-8M | ours: TinyStories-33M | ours: TinyStories-1Layer-21M |
|---|---|---|---|---|---|
| Tom left his ice cream outside in the hot sun. After a little while, … | Says it melted / started to melt or dri… | melt. | melt. | melt. | melt. |
| The room was dark, so Lucy wanted to make it bright. She walked over … | A light, lamp, or light switch. | door. | light. | light. | light. |
| Ben ran and ran until he was very thirsty. He went inside and drank a… | A drink: water, milk, or juice. | water. | water. | water. | water. |
| The little duck lived beside a pond. When she wanted to get to the ot… | Says she swam / began to swim (paddling… | swim around. | struggle. | swim. | swim. |
| Mia looked up at the sky late at night. She saw the big round moon an… | Stars. | clouds. | stars. | stars. | stars. |
| Dad looked out and said it was raining hard. Before Sam went outside,… | Umbrella, raincoat, or boots. | shoes and ran outside. | toys and put them in the bin. | umbrella and opened it. | umbrella and opened it. |
| It got very cold in the winter. The water in the pond outside slowly … | Ice (frozen / froze is acceptable). | the water. | ash. | ice. | a big puddle. |

## Reasoning

| Prompt | Expected | ours: TinyStories-1M | ours: TinyStories-8M | ours: TinyStories-33M | ours: TinyStories-1Layer-21M |
|---|---|---|---|---|---|
| Ben forgot to close the window before the big storm. When he came bac… | Wet (or covered in water / soaked). | gone. | wet. | wet and slippery. | wet. |
| There were three hats on the shelf: a red one, a blue one, and a gree… | The blue hat. | blue one. | green one. | green hat. | red one. |
| Sara had only one coin. The apple cost one coin and the big toy cost … | The apple (the only affordable item). | coin. | apple. | big toy. | toy. |
| Every morning Max brushed his teeth first and ate after that. This mo… | Breakfast (or a breakfast food). | favorite snack. | breakfast. | breakfast. | breakfast. |
| Lily was carrying a full glass of milk much too fast, and then she tr… | Spilled / fell / went everywhere. | was very hot and it was too hot. | spilled all over the floor and made a big mes… | spilled all over her dress and the floor. | spilled all over the floor and made a big mes… |
| It was freezing outside. Tom had a thin coat and a thick coat. He wan… | The thick coat. | coat and coat. | coat and went outside. | coat and the thin coat. | coat. |
| Pip the puppy was very afraid of deep water. His friends ran to the l… | A refusal: no, or that he is scared and… | , "I'm scared of you, I'm scared of you. | yes and jumped in. | , "No, I don't want to. | , "I can't swim, I'm too scared." |

## Context

| Prompt | Expected | ours: TinyStories-1M | ours: TinyStories-8M | ours: TinyStories-33M | ours: TinyStories-1Layer-21M |
|---|---|---|---|---|---|
| Nina and her brother Leo went to the park. Nina brought a red ball an… | The kite (blue kite). | kite. | kite. | kite. | kite. |
| Sam had a toy car and his sister Ella had a toy train. Sam gave his t… | The toy car. | toy car. | toy train. | toy train. | toy car. |
| Mom put the cookies in a tin on the top shelf, and she put the bread … | The tin / the top shelf (not the table … | table. | shelf. | shelf and tried to reach the cookie. | table. |
| The cup on the table was full of juice. Rosa picked it up and drank e… | Empty. | very sweet and yummy. | full and it was time for a nap. | empty. | full of juice. |
| On the farm there were three animals: a cow named Bella, a pig named … | Dot (the duck). | very happy. | very loud. | the perfect match. | very nice. |
| Jon put on his yellow coat and his friend Ana put on her purple coat,… | His yellow coat (yellow, not purple). | coat. | friend. | coat anywhere. | coat. |
