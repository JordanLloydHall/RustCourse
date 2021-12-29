use std::{collections::HashMap, str::EncodeUtf16};

type Label = usize;
type Register = u128;
type State = (Label, HashMap<Register, u128>);

#[derive(Clone, Copy, Debug, PartialEq)]
enum Instruction {
    Add(Register, Label),
    Sub(Register, Label, Label),
    Halt
}

type Godel = u128;

trait Encodable {
    fn encode(&self) -> Godel;
}

impl<T: Encodable> Encodable for &T {
    fn encode(&self) -> Godel {
        (**self).encode()
    }
}

impl Encodable for u128 {
    fn encode(&self) -> Godel {
        *self
    }
}

impl Encodable for i32 {
    fn encode(&self) -> Godel {
        *self as u128
    }
}

impl Encodable for usize {
    fn encode(&self) -> Godel {
        *self as u128
    }
}

impl<T: Encodable, V: Encodable> Encodable for (T, V) {
    fn encode(&self) -> Godel {
        let (x, y) = self;
        (1 << x.encode()) * (2 * y.encode() + 1) - 1
    }
}

impl<T: Encodable> Encodable for Option<T> {
    fn encode(&self) -> Godel {
        match self {
            Some(x) => 1 + x.encode(),
            None => 0
        }
    }
}

impl Encodable for Instruction {
    fn encode(&self) -> Godel {
        match *self {
            Add(i, j) => Some((2 * i, j)).encode(),
            Sub(i, j, k) => Some((2 * i + 1, (j, k))).encode(),
            Halt => (None as Option<u128>).encode()
        }
    }
}

impl<T: Encodable> Encodable for [T] {
    fn encode(&self) -> Godel {
        self.iter()
            .rev()
            .fold(
                (None as Option<u128>).encode(),
                |x, y| Some((y, x)).encode()
            )
    }
}

trait Decodable {
    fn decode(godel: Godel) -> Self;
}

impl Decodable for u128 {
    fn decode(godel: Godel) -> Self {
        godel
    }
}

impl Decodable for usize {
    fn decode(godel: Godel) -> Self {
        godel as usize
    }
}

impl<T: Decodable, V: Decodable> Decodable for (T, V) {
    fn decode(godel: Godel) -> Self {
        let x_godel = godel.trailing_ones() as Godel;
        let y_godel = godel >> (x_godel + 1);

        (decode(x_godel), decode(y_godel))
    }
}

impl<T: Decodable> Decodable for Option<T> {
    fn decode(godel: Godel) -> Self {
        if godel == 0 {
            None
        } else {
            Some(decode(godel - 1))
        }
    }
}

impl Decodable for Instruction {
    fn decode(godel: Godel) -> Self {
        match decode::<Option<(u128, Godel)>>(godel) {
            None => Halt,
            Some((f, arg)) => {
                if f % 2 == 0 {
                    // EVEN: add
                    Add(f / 2, decode(arg))
                } else {
                    // ODD: sub
                    let (j, k) = decode(arg);
                    Sub((f - 1) / 2, j, k)
                }
            }
        }
    }
}

impl<T: Decodable> Decodable for Vec<T> {
    fn decode(mut godel: Godel) -> Self {
        let mut list = vec!();

        while let Some((head, tail)) = decode(godel) {
            list.push(head);

            godel = tail;
        }
    
        list
    }
}

fn decode<T: Decodable>(godel: Godel) -> T {
    T::decode(godel)
}

use Instruction::*;

fn eval_program(program: &[Instruction], init: &State) -> State {
    let (mut label, mut registers) = init.clone();

    loop {
        /* Reveal effects of instruction to state. */
        match program[label] {
            Add(reg, l) => {
                /* Increment target register. */
                *registers.get_mut(&reg).unwrap() += 1;
    
                /* Jump to label. */
                label = l;
            },
            Sub(reg, lsub, lnop) => {
                let register = registers.get_mut(&reg).unwrap();
    
                if *register == 0 {
                    /* Register can't be decremented. */
                    label = lnop;
                } else {
                    /* Decrement register. */
                    *register -= 1;
                    label = lsub;
                }
            },
            Halt => { break; },
        };
    }

    return (label, registers);
}

fn encode_program_to_list(program: &[Instruction]) -> Vec<u128> {
    program.iter().map(Encodable::encode).collect()
}

fn decode_list_to_program(program: &[u128]) -> Vec<Instruction> {
    program.iter().map(|&g| decode(g)).collect()
}

fn main() {
    
}

mod test {
    use crate::*;
    #[test]
    fn godel_num_to_godel_list() {
        let n = 2u128.pow(46) * 20483;
        let godel_list = decode::<Vec<u128>>(n);
        let true_godel_list = vec![46, 0, 10, 1];
        assert_eq!(godel_list, true_godel_list)
    }

    #[test]
    fn godel_list_to_godel_num() {
        let godel_num = (&[46, 0, 10, 1]).encode();
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
            Add(0, 0)
        ];
        let final_state = eval_program(
            &program,
            &(
                0,
                HashMap::<_, _>::from_iter(IntoIter::new([(0, 0), (1, 7)]))
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