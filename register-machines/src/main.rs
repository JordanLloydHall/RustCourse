use std::collections::HashMap;

type Label = usize;
type Register = u128;
type State = (Label, HashMap<Register, u128>);

#[derive(Clone, Copy, Debug, PartialEq)]
enum Instruction {
    Add(Register, Label),
    Sub(Register, Label, Label),
    Halt,
}
use Instruction::*;

#[allow(dead_code)]
fn eval_program(program: &[Instruction], state: &State) -> State {
    let (mut label, mut regs) = state.clone();
    while (0..program.len()).contains(&label) {
        match program[label] {
            Add(reg, new_label) => {
                *regs.entry(reg).or_default() += 1;
                label = new_label;
            }
            Sub(reg, new_label, other_label) => {
                let register = regs.entry(reg).or_default();
                label = if *register > 0 {
                    *register -= 1;
                    new_label
                } else {
                    other_label
                }
            }
            Halt => break,
        };
    }
    (label, regs)
}
// <<x,y>> = (2^x)*(2y+1)
#[allow(dead_code)]
fn encode_pair1(x: u128, y: u128) -> u128 {
    (1 << x) * (2 * y + 1)
}
// <x,y> = (2^x)*(2y+1)-1
#[allow(dead_code)]
fn encode_pair2(x: u128, y: u128) -> u128 {
    encode_pair1(x, y) - 1
}
#[allow(dead_code)]
fn encode_list_to_godel(l: &[u128]) -> u128 {
    match l {
        [] => 0,
        [x, l @ ..] => encode_pair1(*x, encode_list_to_godel(l)),
    }
}
#[allow(dead_code)]
fn encode_program_to_list(program: &[Instruction]) -> Vec<u128> {
    program
        .iter()
        .map(|instruction| match instruction {
            Add(reg, label) => encode_pair1(2 * reg, *label as u128),
            Sub(reg, label, other_label) => encode_pair1(
                2 * reg + 1,
                encode_pair2(*label as u128, *other_label as u128),
            ),
            Halt => 0,
        })
        .collect()
}

// a = (2^x)*(2y+1)
#[allow(dead_code)]
fn decode_pair1(a: u128) -> (u128, u128) {
    let x = a.trailing_zeros() as u128;
    let y = a >> (x + 1);
    (x, y)
}
// a = (2^x)*(2y+1)-1
#[allow(dead_code)]
fn decode_pair2(a: u128) -> (u128, u128) {
    decode_pair1(a + 1)
}
#[allow(dead_code)]
fn decode_godel_to_list(g: u128) -> Vec<u128> {
    fn decode_list_helper(g: u128) -> Vec<u128> {
        if g > 0 {
            let (x, l) = decode_pair1(g);
            let mut vec = decode_list_helper(l);
            vec.push(x);
            vec
        } else {
            [].to_vec()
        }
    }
    let mut vec = decode_list_helper(g);
    vec.reverse();
    vec
}
#[allow(dead_code)]
fn decode_list_to_program(program: &[u128]) -> Vec<Instruction> {
    program
        .iter()
        .map(|encoded| {
            if *encoded == 0 {
                Halt
            } else {
                let (reg, labels) = decode_pair1(*encoded);
                if reg % 2 == 0 {
                    Add(reg / 2, labels as usize)
                } else {
                    let (label, other_label) = decode_pair2(labels);
                    Sub(reg / 2, label as usize, other_label as usize)
                }
            }
        })
        .collect()
}

fn main() {}

mod test {
    use crate::*;
    #[test]
    fn godel_num_to_godel_list() {
        let n = 2u128.pow(46) * 20483;
        let godel_list = decode_godel_to_list(n);
        let true_godel_list = vec![46, 0, 10, 1];
        assert_eq!(godel_list, true_godel_list)
    }

    #[test]
    fn godel_list_to_godel_num() {
        let godel_num = encode_list_to_godel(&[46, 0, 10, 1]);
        assert_eq!(godel_num, 2u128.pow(46) * 20483)
    }

    #[test]
    fn godel_list_to_program() {
        let program = decode_list_to_program(&vec![46, 0, 10, 1]);
        assert_eq!(program, vec![Sub(0, 2, 1), Halt, Sub(0, 0, 1), Add(0, 0)])
    }

    #[test]
    fn program_to_godel_list() {
        let program = encode_program_to_list(&[Sub(0, 2, 1), Halt, Sub(0, 0, 1), Add(0, 0)]);
        assert_eq!(program, [46, 0, 10, 1])
    }

    #[test]
    fn program_produces_correct_state() {
        use std::array::IntoIter;
        let program = vec![
            Sub(1, 2, 1),
            Halt,
            Sub(1, 3, 4),
            Sub(1, 5, 4),
            Halt,
            Add(0, 0),
        ];
        let final_state = eval_program(
            &program,
            &(
                0,
                HashMap::<_, _>::from_iter(IntoIter::new([(0, 0), (1, 7)])),
            ),
        );
        assert_eq!(
            final_state,
            (
                4,
                HashMap::<_, _>::from_iter(IntoIter::new([(0, 2), (1, 0)]))
            )
        )
    }
}
